from loguru import logger
from dataclasses import dataclass
import torch.multiprocessing as mp
import torch
import os
from PIL import Image
import numpy as np
import psutil

from fastmap.container import Images, Tracks


mp.set_start_method("spawn", force=True)  # for CUDA compatibility


def cpu_count() -> int:
    """Get the number of available CPU cores."""
    affinity = psutil.Process().cpu_affinity()
    if affinity is not None:
        return len(affinity)
    else:
        return mp.cpu_count()


@dataclass
class TaskBatch:
    """Batch of data for an image that will be en-queued for consumers."""

    # str, file path of the image
    image_path: str
    # torch.Tensor, long, (num_points2d_for_this_image,), x coordinates of the 2D points in pixels
    x: torch.Tensor
    # torch.Tensor, long, (num_points2d_for_this_image,), y coordinates of the 2D points in pixels
    y: torch.Tensor
    # torch.Tensor, long, (num_points2d_for_this_image,), indices of the 2D points in the color buffer
    point_idx: torch.Tensor


@torch.no_grad()
def consumer_fn(
    color_buffer: torch.Tensor,
    queue: mp.Queue,
):
    """Function for each consumer worker to read the color of 2D points."""
    assert color_buffer.is_shared()
    device = color_buffer.device
    assert color_buffer.dtype == torch.uint8
    while True:
        task_batch = queue.get()
        if task_batch is None:
            break

        # get image path
        image_path = task_batch.image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found.")

        # read image
        im = Image.open(image_path)
        im = im.convert("RGB")
        im = torch.from_numpy(np.array(im)).to(dtype=torch.uint8)  # (H, W, 3)

        # write color for each 2D point
        color_buffer[task_batch.point_idx] = im[
            task_batch.y.cpu(), task_batch.x.cpu()
        ].to(device)
        del im, task_batch


class TrackColor2DReader:
    """Asynchronously read the color of each 2D keypoint from the Tracks container."""

    @torch.no_grad()
    def __init__(
        self,
        tracks: Tracks,
        images: Images,
        image_dir: str,
        use_cpu: bool = True,
    ):
        self.num_workers = cpu_count() - 1
        self.num_workers = max(1, self.num_workers)  # at least one worker
        logger.info(f"Using {self.num_workers} workers for color reading.")
        self.tracks = tracks
        self.images = images
        self.image_dir = image_dir

        # set the device for buffer and output
        self.buffer_device = torch.device("cpu") if use_cpu else tracks.device
        self.output_device = tracks.device
        logger.info(
            f"Using buffer device {self.buffer_device} and output device {self.output_device}."
        )

        # make sure the image_dir exists
        assert self.image_dir is not None
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} not found.")

        # create the job queue
        self.queue = mp.Queue()

        # create the buffer for color for 2D points
        self.color2d = torch.zeros(
            tracks.num_points2d, 3, device=self.buffer_device, dtype=torch.uint8
        )  # (num_points2d, 3)
        self.color2d.share_memory_()  # share memory for all workers

        # create the consumers
        self.consumers = []
        for _ in range(self.num_workers):
            worker = mp.Process(
                target=consumer_fn,
                args=(self.color2d, self.queue),
            )
            worker.start()
            self.consumers.append(worker)

    @torch.no_grad()
    def start(self):
        """Populate the job queue."""

        # populate the job queue with tasks
        logger.info("Populating job queue...")
        for image_idx, image_name in enumerate(self.images.names):
            # get image path
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")

            # get the 2D point indices for this image
            point_mask = self.tracks.image_idx == image_idx  # (num_points2d, )
            if not torch.any(point_mask):
                continue
            point_idx = point_mask.nonzero(as_tuple=False).squeeze(
                1
            )  # (num_points2d_for_image, )
            del point_mask

            # get the 2D point coordinates in pixels
            xy = self.tracks.xy_pixels[point_idx]  # (num_points2d_for_image, 2)
            x, y = xy.unbind(-1)  # (num_points2d_for_image,), (num_points2d_for_image,)
            x = x.clamp(
                0, self.images.widths[image_idx].item() - 1
            ).long()  # (num_points2d_for_image,)
            y = y.clamp(
                0, self.images.heights[image_idx].item() - 1
            ).long()  # (num_points2d_for_image,)
            assert torch.all(x >= 0) and torch.all(x < self.images.widths[image_idx])
            assert torch.all(y >= 0) and torch.all(y < self.images.heights[image_idx])
            del xy

            # build the task batch
            task_batch = TaskBatch(
                image_path=image_path,
                x=x.to(self.buffer_device),
                y=y.to(self.buffer_device),
                point_idx=point_idx.to(self.buffer_device),
            )
            self.queue.put(task_batch)

        # put None to signal the end of the queue
        for _ in range(self.num_workers):
            self.queue.put(None)

        # log
        logger.info("Job queue populated.")

    @torch.no_grad()
    def join(self):
        """Join the consumers and return the color for 2D points."""
        for worker in self.consumers:
            worker.join()
        return self.color2d.to(self.output_device)
