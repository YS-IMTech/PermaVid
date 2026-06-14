import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
from utils.camera_convert import quaternion_to_w2c, poses_intrinsics_to_coordinates, read_uepose_from_json, euler_to_w2c_batch
import numpy as np


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image


    def resize(self, image, target_height, target_width):
        width, height = image.size
        image = torchvision.transforms.functional.resize(
            image,
            (target_height, target_width),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.resize(data, *self.get_height_width(data))

        # image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]



class Loadcoord(DataProcessingOperator):
    def __init__(self, 
                 num_frames=81,
                 intrinsic = [0.5, 0.8667, 0.5, 0.5],
                 orig_intrinsic=False,
                 max_length=1000):
        
        self.intrinsic = intrinsic
        self.orig_intrinsic = orig_intrinsic
        self.num_frames = num_frames
        self.max_length = max_length
        
    def __call__(self, data: str):


        if data.endswith('.json'): ## uedata (poses)
            print("It's a JSON file")
            poses = euler_to_w2c_batch(read_uepose_from_json(data), return_34=True)
            num_frames = poses.shape[0]
            intrinsics = np.tile(np.array(self.intrinsic, dtype=np.float32), (num_frames, 1)) 
            coordinates = poses_intrinsics_to_coordinates(w2c_poses=poses, intrinsics=intrinsics)
            if len(coordinates) > self.max_length:
                coordinates = coordinates[:self.max_length]
            
        elif data.endswith('.npz'): ## sekai (poses)
            print("It's an NPZ file")
            poses_data = np.load(data)
            poses = poses_data['extrinsic']
            # poses[:, :3, 3] = poses[:, :3, 3] * 2
            poses = poses[:, :3, :4]
            K = poses_data['intrinsic']
            raw_num_frames = poses.shape[0]
            intrinsics = np.tile(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32), (raw_num_frames, 1)) 
            coordinates = poses_intrinsics_to_coordinates(w2c_poses=poses, intrinsics=intrinsics)
            if len(coordinates) > self.max_length:
                coordinates = coordinates[:self.max_length]
                
        else: # spatialvid (poses)
            poses = quaternion_to_w2c(data)
            num_frames_to_read = poses.shape[0]

            if num_frames_to_read >= self.num_frames:
                num_frames = self.num_frames
            elif num_frames_to_read < self.num_frames:
                num_frames = ((num_frames_to_read - 1) // 4) * 4 + 1    

            poses = poses[:num_frames]
            
            if self.orig_intrinsic:
                intrinsics_path = data.replace("poses.npy", "intrinsics.npy")
                intrinsics = np.load(intrinsics_path)
                intrinsics = intrinsics[:num_frames]
            else:
                intrinsics = np.tile(np.array(self.intrinsic, dtype=np.float32), (num_frames, 1)) 
            coordinates = poses_intrinsics_to_coordinates(w2c_poses=poses, intrinsics=intrinsics)            
            if len(coordinates) > self.max_length:
                coordinates = coordinates[:self.max_length]

        return coordinates



class LoadPosejson2coord(DataProcessingOperator):
    def __init__(self, 
                 intrinsic = [0.5, 0.8667, 0.5, 0.5],
                 orig_intrinsic=False):
        
        self.intrinsic = intrinsic
        self.orig_intrinsic = orig_intrinsic
        
    def __call__(self, data: str):
        poses = euler_to_w2c_batch(read_uepose_from_json(data), return_34=True)
        num_frames = poses.shape[0]
        intrinsics = np.tile(np.array(self.intrinsic, dtype=np.float32), (num_frames, 1)) 
        coordinates = poses_intrinsics_to_coordinates(w2c_poses=poses, intrinsics=intrinsics)

        return coordinates





class LoadPose2coord(DataProcessingOperator):
    def __init__(self, 
                 num_frames=81,
                 intrinsic = [0.5, 0.8667, 0.5, 0.5],
                 orig_intrinsic=True):
        
        self.intrinsic = intrinsic
        self.num_frames = num_frames
        self.orig_intrinsic = orig_intrinsic
        
    def __call__(self, data: str):
        poses = quaternion_to_w2c(data)
        num_frames_to_read = poses.shape[0]

        if num_frames_to_read >= self.num_frames:
            num_frames = self.num_frames
        elif num_frames_to_read < self.num_frames:
            num_frames = ((num_frames_to_read - 1) // 4) * 4 + 1    

        poses = poses[:num_frames]
        
        if self.orig_intrinsic:
            intrinsics_path = data.replace("poses.npy", "intrinsics.npy")
            intrinsics = np.load(intrinsics_path)
            intrinsics = intrinsics[:num_frames]
        else:
            intrinsics = np.tile(np.array(self.intrinsic, dtype=np.float32), (num_frames, 1)) 
        coordinates = poses_intrinsics_to_coordinates(w2c_poses=poses, intrinsics=intrinsics)

        return coordinates


    

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is built in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        total = int(reader.count_frames())
        if total <= 0:
            raise ValueError("Video has no frames.")

        return total
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        frames = []
        try:
            num_frames = self.get_num_frames(reader)
            for frame_id in range(num_frames):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
        
            
            return frames
        finally:
            reader.close()




class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])

    # @staticmethod
    # def default_video_operator(
    #     base_path="",
    #     max_pixels=1920*1080, height=None, width=None,
    #     height_division_factor=16, width_division_factor=16,
    #     num_frames=81, time_division_factor=4, time_division_remainder=1,
    # ):
    #     return RouteByType(operator_map=[
    #         (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
    #             (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
    #             (("gif",), LoadGIF(
    #                 num_frames, time_division_factor, time_division_remainder,
    #                 frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
    #             )),
    #             (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
    #                 num_frames, time_division_factor, time_division_remainder,
    #                 frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
    #             )),
    #         ])),
    #     ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        try:
            if self.load_from_cache:
                path = self.cached_data[data_id % len(self.cached_data)]
                data = self.cached_data_operator(path)
            else:
                raw_data = self.data[data_id % len(self.data)].copy()
                data = raw_data.copy()
                for key in self.data_file_keys:
                    if key in data:
                        try:
                            if key in self.special_operator_map:
                                data[key] = self.special_operator_map[key](data[key])
                            else:
                                data[key] = self.main_data_operator(data[key])
                        except Exception as e:
                            print(f"⚠️ Failed to process key '{key}' in sample {data_id}: {e}")
                            return None 
            return data
        except Exception as e:
            print(f"⚠️ Failed to load sample {data_id}: {e}")
            return None  
        
        
    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
