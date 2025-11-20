import numpy as np
import os
import shutil
import tempfile
from dataclasses import dataclass
import time
import gc


# --- 1. DATACLASS FOR METADATA ---

@dataclass(frozen=True)
class TextureMetadata:
    """
    Stores the necessary information for a worker process to access
    the shared texture data stored on disk via memory mapping.

    global_dtype: The shared NumPy dtype for all textures (e.g., 'uint8').
    texture_infos: list of dicts, where each dict contains 'shape',
                   and 'offset' (in bytes) for the raw data of an individual texture.
    """
    filepath: str
    global_dtype: str
    global_number_of_channels: int
    texture_infos: list[dict[str, object]]
    total_bytes: int


# --- 2. UTILITY FUNCTIONS FOR WORKERS (Unchanged core logic) ---

# Cache for the 1D memory-mapped data array in each worker process
_shared_memmap_cache: dict[str, np.memmap] = {}  # Using native dict generic


def get_base_memmap(metadata: TextureMetadata) -> np.memmap:
    """
    Retrieves the 1D, raw byte memory map for the data file, caching it
    within the worker process for quick access.
    """
    filepath = metadata.filepath

    if filepath not in _shared_memmap_cache:
        try:
            # Create a 1D memmap view of the raw bytes
            # dtype='uint8' is chosen because we are dealing with raw bytes
            shared_array = np.memmap(
                filename=filepath,
                dtype=np.uint8,
                mode='r',
                shape=(metadata.total_bytes,)
            )
            _shared_memmap_cache[filepath] = shared_array
        except FileNotFoundError:
            print(f"Worker {os.getpid()}: ERROR - Shared file not found at {filepath}")
            raise

    return _shared_memmap_cache[filepath]


def get_individual_texture(metadata: TextureMetadata, index: int) -> np.ndarray:
    """
    Accesses the raw byte data for the texture at the given index from the
    shared memory map, then casts and reshapes it to the correct dimensions.
    """
    if index >= len(metadata.texture_infos):
        raise IndexError("Texture index out of bounds.")

    info = metadata.texture_infos[index]

    # 1. Get the shared 1D raw byte view (cached)
    base_map = get_base_memmap(metadata)

    # 2. Get the global dtype and related information
    element_dtype = np.dtype(metadata.global_dtype)

    start_byte = info['offset']
    end_byte = start_byte + info['nbytes']

    # Note: start_index and end_index are byte positions since base_map is uint8
    start_index = start_byte
    end_index = end_byte

    # 3. Extract the raw byte slice for the texture
    raw_slice = base_map[start_index:end_index]

    # 4. Cast the raw bytes to the correct dtype and reshape (zero-copy view)
    texture = raw_slice.view(dtype=element_dtype).reshape(tuple(info['shape']))

    return texture


# --- 3. SEQUENCE PROTOCOL WRAPPER (Public Interface) ---

class SharedTextureList:
    """
    The main class for creating, managing, and accessing shared, variable-sized
    NumPy arrays via a single memory-mapped file.

    It implements the sequence protocol (__len__, __getitem__) to allow use
    of list syntax (e.g., texture_list[i]).
    """

    def __init__(self, metadata: TextureMetadata):
        """Private constructor. Use SharedTextureList.from_list() instead."""
        self.metadata = metadata

        # Pre-prime the base map cache in the current process
        try:
            get_base_memmap(self.metadata)
        except Exception as e:
            print(f"ERROR: SharedTextureList failed to initialize base map: {e}")

    @classmethod
    def from_list(cls, texture_list: list[np.ndarray]) -> 'SharedTextureList':
        """
        Alternative constructor that creates the shared file on disk and
        returns a SharedTextureList instance to access it.
        """
        if not texture_list:
            raise ValueError("Texture list cannot be empty.")

        for texture in texture_list:
            if len(texture.shape) <= 2:
                raise ValueError("All textures must have shape equal to 3.")

        # 1. Validate and extract the common dtype & number of channels
        first_dtype = texture_list[0].dtype
        for texture in texture_list:
            if texture.dtype != first_dtype:
                raise ValueError("All textures must have the same dtype.")

        first_numb_channels = texture_list[0].shape[2]
        for texture in texture_list:
            if texture.shape[2] != first_numb_channels:
                raise ValueError("All textures must have the same number of channels.")

        base_dtype_str = str(first_dtype)
        number_of_channels = first_numb_channels

        # 2. Setup file path in a secure, OS-appropriate temporary location
        temp_dir = tempfile.mkdtemp(prefix='texture_synth_')
        filename = os.path.join(temp_dir, f'shared_textures_{time.time_ns()}.dat')

        texture_infos: list[dict[str, object]] = []
        current_offset_bytes = 0
        total_bytes_written = 0

        print(f"SharedTextureList: Writing raw data to {filename}...")

        # 3. Write all raw data sequentially to the file
        with open(filename, 'wb') as f:
            for i, texture in enumerate(texture_list):
                texture = np.ascontiguousarray(texture)
                texture_bytes = texture.tobytes()
                byte_size = len(texture_bytes)

                f.write(texture_bytes)

                texture_infos.append({
                    'shape': list(texture.shape),
                    'offset': current_offset_bytes,
                    'nbytes': byte_size
                })

                current_offset_bytes += byte_size
                total_bytes_written += byte_size

        print(f"SharedTextureList: File created with total size {total_bytes_written / (1024 * 1024):.2f} MB.")

        # 4. Create metadata and initialize the instance
        metadata = TextureMetadata(
            filepath=filename,
            global_dtype=base_dtype_str,
            global_number_of_channels=number_of_channels,
            texture_infos=texture_infos,
            total_bytes=total_bytes_written
        )
        return cls(metadata)

    def release(self):
        """
        Deletes the shared memory-mapped file and its containing temporary directory.
        This must be called when the data is no longer needed.
        """
        # Get the directory (which is the temp directory we created)
        temp_dir = os.path.dirname(self.metadata.filepath)

        # In a real-world multi-process scenario, cleanup is safest *after* # all worker processes have exited. This cleanup method is designed
        # to be called by the main process once the Parallel job is done.
        try:
            # Explicitly remove the memmap reference from the cache in the current process.
            filepath = self.metadata.filepath
            if filepath in _shared_memmap_cache:
                # Drop the reference.
                del _shared_memmap_cache[filepath]

                # Forcing a collection helps ensure the file handle is released immediately.
                gc.collect()
                print(f"SharedTextureList: Released memmap file handle for {filepath}.")

            shutil.rmtree(temp_dir)
            print(f"SharedTextureList: Cleaned up temporary directory and shared file: {temp_dir}")
        except OSError as e:
            print(f"SharedTextureList: WARNING - Could not remove temporary directory {temp_dir}: {e}")

    def __len__(self) -> int:
        """Enables len(texture_list)"""
        return len(self.metadata.texture_infos)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Enables texture_list[i] notation.
        This dynamically fetches the texture from the shared memory map.
        """
        if isinstance(index, slice):
            raise NotImplementedError("Slicing (e.g., texture_list[:5]) is not supported.")

        return get_individual_texture(self.metadata, index)

    # --- Context Manager Implementation ---
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        # Returning None means we don't suppress any exceptions that occurred
