import torch, os, json
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser
from diffsynth.trainers.unified_dataset_ue import UnifiedDataset, LoadVideo, ImageCropAndResize, ToAbsolutePath, LoadPose2coord, LoadPosejson2coord
from diffsynth.models.wan_video_camera_controller import SimpleAdapter
import wandb
import random
import numpy as np
from PIL import Image
from utils.utils_keyframe import extract_keyframes_indices
os.environ["TOKENIZERS_PARALLELISM"] = "false"



class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, 
        resume_from_checkpoint=None,
        remove_prefix=None,
        add_control_adapter=False,
        train_mode = "onlycam",     #["onlycam", "memo_rgb", "memo_mix"]
        reference_nums = 5,           
        model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, 
                                                     device="cpu", 
                                                     redirect_common_files=False,
                                                     model_configs=model_configs)
        
        
        if add_control_adapter:
            self.pipe.dit.add_control_adapter = True
            self.pipe.dit.control_adapter = SimpleAdapter(self.pipe.dit.in_dim_control_adapter, self.pipe.dit.dim, kernel_size=self.pipe.dit.patch_size[1:], stride=self.pipe.dit.patch_size[1:])

        self.train_mode = train_mode
        self.reference_nums = reference_nums
        
        ## Load resume ckpt
        if resume_from_checkpoint:
            state_dict_ = {}
            resume_state_dict = load_state_dict(resume_from_checkpoint, torch_dtype=torch.bfloat16, device="cpu")
            if isinstance(remove_prefix, str):
                remove_prefix = [prefix.strip() for prefix in remove_prefix.split(",")]

            for name, param in resume_state_dict.items():
                for prefix in remove_prefix:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                state_dict_[name] = param
                
            resume_state_dict = state_dict_
            
            vace_state_dict = {}
            dit_state_dict = {}
            for k, v in resume_state_dict.items():
                if "vace" in k:
                    vace_state_dict[k] = v
                else:
                    dit_state_dict[k] = v

            if vace_state_dict:
                self.pipe.vace.load_state_dict(vace_state_dict, strict=True)
                print(f"[INFO] Loaded {len(vace_state_dict)} VACE parameters from: {resume_from_checkpoint}")
            if dit_state_dict:
                self.pipe.dit.load_state_dict(dit_state_dict, strict=True)
                print(f"[INFO] Loaded {len(dit_state_dict)} DiT parameters from: {resume_from_checkpoint}")

            del resume_state_dict, vace_state_dict, dit_state_dict

 
    
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # ## force dit text embedding grad false
        # for name, param in  self.pipe.dit.named_parameters():

        #     for not_trainable_name in ['text_embedding', 'time_embedding', 'time_projection']:
        #         if not_trainable_name in name:
        #             param.requires_grad = False
        #             print("name grad false:", name)         
    
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        
        ################################### add ###################################
        raw_num_frames = len(data["video"])

        num_frames_candidates = list(range(25, 73+1, 4))  # 25, 29, ..., 81
        num_frames = random.choice(num_frames_candidates)

        # self.train_mode = random.choices(
        #     population=['memo_rgb', 'memo_mix'],
        #     weights=[0.5, 0.5],
        #     k=1
        # )[0]
        print("train mode:", self.train_mode)
        
        print("gen num_frames:", num_frames, "raw_num_frames:", raw_num_frames)
        
        if self.train_mode == "onlycam": #["onlycam", "memo_rgb", "memo_mix"]
            gen_start_idx = random.randint(0, raw_num_frames - num_frames - 1)
            gen_end_idx = gen_start_idx + num_frames
            
            memory_start_idx = None
            memory_end_idx = None
            # print("gen_start_idx:", gen_start_idx)
            # print("gen_end_idx:", gen_end_idx)


        elif self.train_mode == "memo_rgb":
            gen_start_idx = random.randint(100, raw_num_frames - num_frames - 1)
            gen_end_idx = gen_start_idx + num_frames

            memory_start_idx = 0
            memory_end_idx = gen_start_idx 

            # print("gen_start_idx:", gen_start_idx)
            # print("gen_end_idx:", gen_end_idx)
            # print("memory_start_idx:", memory_start_idx)
            # print("memory_end_idx:", memory_end_idx)
            
        elif self.train_mode == "memo_mix":
            gen_start_idx = random.randint(100, raw_num_frames - num_frames - 1)
            gen_end_idx = gen_start_idx + num_frames

            memory_start_idx = 0
            memory_end_idx = gen_start_idx 
            

     
        if memory_start_idx is not None and memory_end_idx is not None:
            
            # overlap_threshold = random.choice([0.4, 0.5, 0.6, 0.7, 0.8])
            overlap_threshold = 0.4
            
            keyframes_indices = extract_keyframes_indices(coordinates = data["poses"], 
                                                        reference_nums = self.reference_nums,
                                                        memory_start = memory_start_idx,
                                                        memory_end = memory_end_idx,
                                                        traj_start = gen_start_idx,
                                                        traj_end = gen_end_idx,
                                                        height = data["video"][0].size[1], 
                                                        width = data["video"][0].size[0],
                                                        overlap_threshold=overlap_threshold,
                                                        fast=True)
        #["onlycam", "memo_rgb", "memo_mix"]
        vace_reference_image = None
            
        if self.train_mode == "memo_rgb" and keyframes_indices is not None:
            keyframes_indices = sorted(keyframes_indices)
            vace_reference_image = []
            for index in keyframes_indices:
                vace_reference_image.append(data["video"][index])
            
            
        elif self.train_mode == "memo_mix" and keyframes_indices is not None:
            vace_reference_image = []
            
            keyframes_indices = sorted(keyframes_indices)
            k_reference = len(keyframes_indices)
            
            
            # if random.random() < 0.5:
            #     k_reference_depth = random.randint(1, k_reference)
            # else:
            #     k_reference_depth = k_reference
            
            ## only depth reference memory
            k_reference_depth = k_reference
            
            keyframes_indices_depth = set(random.sample(keyframes_indices, k_reference_depth))
            keyframes_indices_rgb = [x for x in keyframes_indices if x not in keyframes_indices_depth]

            keyframes_indices_depth = sorted(list(keyframes_indices_depth))
            keyframes_indices_rgb = sorted(keyframes_indices_rgb)

            print("------------------------------------------------")
            print("keyframes_indices:", keyframes_indices)
            print("k_reference_depth:", k_reference_depth)
            print("keyframes_indices_depth:", keyframes_indices_depth)
            print("keyframes_indices_rgb:", keyframes_indices_rgb)
            print("------------------------------------------------")
            
            for index in keyframes_indices:
                if keyframes_indices_rgb and index in keyframes_indices_rgb:
                    vace_reference_image.append(data["video"][index])
                if keyframes_indices_depth and index in keyframes_indices_depth:
                    vace_reference_image.append(data["depth"][index])
                    
                
                
        data["video"] = data["video"][gen_start_idx:gen_end_idx] 
        data["coordinates"] = data["poses"][gen_start_idx:gen_end_idx] 
        height, width = data["video"][0].size[1], data["video"][0].size[0]
        gray = np.full((height, width, 3), 127, dtype=np.uint8)
        data["vace_video"] = [data["video"][0]] + [Image.fromarray(gray)] * (num_frames - 1)  

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # "original_pose_height": 480,
            # "original_pose_width": 832,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        if vace_reference_image is not None:
            inputs_shared["vace_reference_image"] = vace_reference_image
        
        if data["coordinates"] is not None:
            inputs_shared["coordinates"] = data["coordinates"]

        if data["vace_video"] is not None:
            inputs_shared["vace_video"] = data["vace_video"]

        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            # elif extra_input == "poses":
            #     inputs_shared["coordinates"] = data[extra_input]
            elif extra_input == "original_pose_height" or extra_input == "original_pose_width":
                inputs_shared[extra_input] = int(data[extra_input])
            else:
                inputs_shared[extra_input] = data[extra_input]


        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        # print("================================ training inputs_shared params =====================================")
        
        # print("inputs_shared:", inputs_shared.keys()) # add new ['noise', 'latents', 'vace_context']    

    
        return {**inputs_shared, **inputs_posi}
    
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()

    parser.add_argument("--add_control_adapter", default=False, action="store_true")
    parser.add_argument("--train_mode", type=str, default='False',
                        choices=["onlycam", "memo_rgb", "memo_mix"],)

    parser.add_argument("--reference_nums", type=int, default=10)

    # ==== add：wandb login & init (before dataset/model creation) ====
    parser.add_argument("--wandb_host", type=str, default="https://api.wandb.ai", help="Wandb API host.")
    parser.add_argument("--wandb_key", type=str, default="4261e7907c4365cc1feb874945c9d718fdb765cf", help="Wandb API key.")
    parser.add_argument("--wandb_entity", type=str, default="yssssmikey", help="Wandb entity (team/user).")
    parser.add_argument("--wandb_project", type=str, default="causalworld", help="Wandb project name.")
    parser.add_argument("--wandb_name", type=str, default="causalworld_camctrl_noref_1.3b", help="Wandb run name.")
    parser.add_argument("--wandb_log_interval", type=int, default=1, help="Log to wandb every N steps.")

    args = parser.parse_args()

    # ==== Initialize wandb (even before accelerator) ====
    try:
        if args.wandb_host and args.wandb_host != "https://api.wandb.ai":
            os.environ["WANDB_BASE_URL"] = args.wandb_host
        if args.wandb_key:
            wandb.login(key=args.wandb_key)
    except Exception as e:
        print(f"⚠️ wandb init failed: {e}")
        args.wandb_project = None
    # ==== end wandb setup ====


    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "poses": ToAbsolutePath(args.dataset_base_path) >> LoadPosejson2coord(intrinsic = [0.5, 0.8667, 0.5, 0.5],
                                                                                orig_intrinsic=True)

            # "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        resume_from_checkpoint = args.resume_from_checkpoint, ##add
        remove_prefix = args.remove_prefix_in_ckpt,           ##add
        add_control_adapter = args.add_control_adapter,       ##add
        train_mode = args.train_mode,                         ##add
        reference_nums = args.reference_nums,                 ##add
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    launch_training_task(dataset, model, model_logger, args=args)
