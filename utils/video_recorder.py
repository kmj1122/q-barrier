import cv2
import numpy as np
import os
import sys
from PIL import Image
import imageio
from typing import List, Optional, Union
import threading

class VideoRecorder:
    """Universal video recorder for gymnasium environments."""
    
    def __init__(self, 
                 output_dir,
                 video_name,
                 fps: int = 30,
                 save_individual_frames: bool = False):
        """
        Initialize video recorder.
        
        Args:
            output_dir: Directory to save videos
            video_name: Base name for video files
            fps: Frames per second for video
            save_individual_frames: Whether to save individual frame images
        """
        self.output_dir = output_dir
        self.video_name = video_name
        self.fps = fps
        self.save_individual_frames = save_individual_frames
        self.frames = []
        self._save_threads = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        if save_individual_frames:
            self.frames_dir = os.path.join(output_dir, f"{video_name}_frames")
            os.makedirs(self.frames_dir, exist_ok=True)
    
    def add_frame(self, frame: Union[np.ndarray, tuple]) -> None:
        """Add a frame to the recording."""
        if frame is not None:
            # Handle case where render() returns a tuple (frame, info)
            if isinstance(frame, tuple):
                frame = frame[0]  # Extract the actual frame from tuple
            
            # Skip if frame is still None after extraction
            if frame is None:
                return
                
            # Ensure frame is numpy array
            if not isinstance(frame, np.ndarray):
                return
            
            # Ensure frame is in correct format (H, W, 3) with uint8 dtype
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            self.frames.append(frame.copy())
            
            # Save individual frame if requested
            if self.save_individual_frames:
                frame_path = os.path.join(self.frames_dir, f"frame_{len(self.frames):06d}.png")
                Image.fromarray(frame).save(frame_path)
    
    def save_mp4(self, filename: Optional[str] = None) -> str:
        """Start saving recorded frames as MP4 video in a background thread and return immediately.

        Returns the target video path. Use `wait_for_saves()` if you need to block
        until background saves finish.
        """
        if not self.frames:
            raise ValueError("No frames recorded!")

        if filename is None:
            filename = f"{self.video_name}.mp4"

        video_path = os.path.join(self.output_dir, filename)

        # Copy frames so background process is independent of future mutations
        frames_to_save = [frame.copy() for frame in self.frames]

        def _write_mp4(frames: List[np.ndarray], fps: int, path: str) -> None:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, fps, (width, height))
            for frame in frames:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            out.release()
            print(f"MP4 video saved: {path}")

        thread = threading.Thread(target=_write_mp4, args=(frames_to_save, self.fps, video_path))
        thread.start()
        self._save_threads.append(thread)
        return video_path
    
    def save_gif(self, filename: Optional[str] = None, duration: float = None) -> str:
        """Start saving recorded frames as GIF in a background thread and return immediately."""
        if not self.frames:
            raise ValueError("No frames recorded!")

        if filename is None:
            filename = f"{self.video_name}.gif"

        if duration is None:
            duration = 1000 // self.fps  # Convert fps to milliseconds per frame

        gif_path = os.path.join(self.output_dir, filename)

        frames_to_save = [frame.copy() for frame in self.frames]

        def _write_gif(frames: List[np.ndarray], duration_ms: float, path: str) -> None:
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"GIF saved: {path}")

        thread = threading.Thread(target=_write_gif, args=(frames_to_save, duration, gif_path))
        thread.start()
        self._save_threads.append(thread)
        return gif_path
    
    def save_webm(self, filename: Optional[str] = None) -> str:
        """Start saving recorded frames as WebM video using imageio in a background thread and return immediately."""
        if not self.frames:
            raise ValueError("No frames recorded!")

        if filename is None:
            filename = f"{self.video_name}.webm"

        video_path = os.path.join(self.output_dir, filename)

        frames_to_save = [frame.copy() for frame in self.frames]

        def _write_webm(frames: List[np.ndarray], fps: int, path: str) -> None:
            with imageio.get_writer(path, fps=fps, codec='libvpx-vp9') as writer:
                for frame in frames:
                    writer.append_data(frame)
            print(f"WebM video saved: {path}")

        thread = threading.Thread(target=_write_webm, args=(frames_to_save, self.fps, video_path))
        thread.start()
        self._save_threads.append(thread)
        return video_path
    
    def save_all_formats(self) -> dict:
        """Start saving video in all supported formats in background threads and return immediately with target paths."""
        results = {}

        try:
            results['mp4'] = self.save_mp4()
        except Exception as e:
            print(f"Failed to start MP4 save: {e}")

        try:
            results['gif'] = self.save_gif()
        except Exception as e:
            print(f"Failed to start GIF save: {e}")

        try:
            results['webm'] = self.save_webm()
        except Exception as e:
            print(f"Failed to start WebM save: {e}")

        return results
    
    def clear_frames(self) -> None:
        """Clear all recorded frames."""
        self.frames.clear()
    
    def get_frame_count(self) -> int:
        """Get number of recorded frames."""
        return len(self.frames)
    
    def get_duration(self) -> float:
        """Get video duration in seconds."""
        return len(self.frames) / self.fps if self.frames else 0.0

    def wait_for_saves(self, timeout: Optional[float] = None) -> None:
        """Block until all background save threads finish or an optional timeout elapses.

        If a timeout is provided, the method returns after the timeout and
        leaves any still-running threads active.
        """
        remaining = []
        for thread in self._save_threads:
            thread.join(timeout=timeout)
            if thread.is_alive():
                remaining.append(thread)
        self._save_threads = remaining