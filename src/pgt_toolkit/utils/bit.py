# Import external modules
import numpy as np


# --------------------------------------------
# Binary List Manipulation Functions
# --------------------------------------------
class BitUtils:
    """
    A collection of static methods for working with binary lists (bit patterns) and
    converting them to and from integer IDs.
    """

    # --- Bits and ID conversion functions  ---
    @staticmethod
    def list_shift_left(lst: list, k: int) -> list:
        """
        Shifts the elements of a list cyclically to the left (rotate left operation).

        Args:
            lst (list): The input list (e.g., a bit pattern).
            k (int): The number of positions to shift left.

        Returns:
            list: The cyclically rotated list.
        """
        # Ensure k is within the bounds of the list length for efficient rotation.
        k = k % len(lst)
        return lst[k:] + lst[:k]

    @staticmethod
    def flip_bits(code_list: list) -> list:
        """
        Inverts the bit values in the code list: 0 becomes 1, and 1 becomes 0.

        Args:
            code_list (list): The binary list (e.g., bit pattern) to invert.

        Returns:
            list: The bit-flipped list.
        """
        return [1 - i for i in code_list]

    @staticmethod
    def id_to_bits(id_value: int, N: int) -> list:
        """
        Converts an integer ID into its canonical bit list representation.

        This first converts the integer to a raw bit list, then converts it back
        to its canonical ID, and finally converts the canonical ID back to bits
        to ensure the resulting bit list is one of the valid rotations for that ID.

        Args:
            id_value (int): The integer ID.
            N (int): The length of the bit list.

        Returns:
            list: The canonical bit list representation (LSB at index 0).
        """
        # The function logic implies a specific process for canonical bit list
        # generation, likely to ensure the smallest rotation ID is used.
        bits = BitUtils.int_to_bits(id_value, N)
        id_new = BitUtils.bits_to_id(bits, N)
        bits_new = BitUtils.int_to_bits(id_new, N)
        return bits_new

    @staticmethod
    def int_to_bits(int_value: int, N: int) -> list:
        """
        Converts a raw integer to a binary list of length N (Least Significant Bit at index 0).

        Args:
            int_value (int): The integer ID to convert.
            N (int): The desired length of the bit list.

        Returns:
            list: The binary list representation (LSB at index 0).
        """
        bits = [0] * N
        # Create a temporary copy of value to modify in the loop
        temp_value = int_value
        for i in range(0, N):
            if temp_value > 0:
                bits[i] = temp_value % 2
                temp_value = temp_value // 2
            # The 'else' case is implicitly handled by the initialization 'bits = [0] * N'
        return bits

    @staticmethod
    def bits_to_id(bits: list, N: int) -> int:
        """
        Converts a binary list (LSB first) to the canonical integer ID.

        This is achieved by checking all N possible cyclic rotations of the bit pattern
        and returning the smallest integer value found (Canonicalization).

        Args:
            bits (list): The input binary list (e.g., a decoded pattern).
            N (int): The length of the bit list.

        Returns:
            int: The canonical (smallest rotational) integer ID.
        """
        min_int_value = float("inf")
        # Use a copy to ensure the input list 'bits' is not modified
        curr_bits = bits[:]

        # Iterate through all N possible rotations
        for _ in range(0, N):
            # Calculate the raw integer value for the current rotation
            temp_int_value = BitUtils.bits_to_int(curr_bits, N)

            # Track the minimum integer value (the canonical ID)
            min_int_value = min(temp_int_value, min_int_value)

            # Rotate the list one position to the left for the next iteration
            curr_bits = BitUtils.list_shift_left(curr_bits, 1)

        return min_int_value

    @staticmethod
    def bits_to_int(bits: list, N: int) -> int:
        """
        Converts a binary list (LSB first) to its raw integer value.

        Args:
            bits (list): The input binary list (LSB at index 0).
            N (int): The length of the bit list.

        Returns:
            int: The raw integer value.
        """
        int_value = 0
        for j in range(0, N):
            # Calculate integer value assuming LSB (2^j) is at index j.
            if bits[j] == 1:
                int_value = int_value + 2**j
        return int_value

    # --- Alternating pattern functions ---
    @staticmethod
    def _clean_noise(sampled_values: np.ndarray) -> np.ndarray:
        """
        STEP 1: Cleans single-pixel noise (spikes) using a local 3-window
        circular majority filter.

        It identifies and corrects isolated pixels that differ from both
        their left and right neighbors.
        Example: [1, 1, 0, 1, 1] -> [1, 1, 1, 1, 1] (0-spike corrected to 1)

        Args:
            sampled_values: A 1D NumPy array of binary pixel values.

        Returns:
            np.ndarray: The noise-cleaned array.
        """
        if sampled_values.size < 3:
            return sampled_values  # Cannot clean noise on arrays < 3

        # Use cyclic rolling to get circular neighbors
        left_neighbors = np.roll(sampled_values, 1)
        right_neighbors = np.roll(sampled_values, -1)

        # Identify positions where a pixel is a 'spike':
        # 1. The left neighbor equals the right neighbor (i.e., [A, X, A])
        # 2. The left neighbor is NOT equal to the current pixel (i.e., X != A)
        # This guarantees the center pixel (X) is isolated noise.
        is_spike = np.logical_and(
            left_neighbors == right_neighbors, sampled_values != left_neighbors
        )

        # Create the cleaned array: replace spikes with the surrounding neighbor value
        cleaned_values = np.where(is_spike, left_neighbors, sampled_values)

        return cleaned_values

    @staticmethod
    def _align_pattern(sampled_values: np.ndarray) -> np.ndarray:
        """
        STEP 2: Cyclically shifts the pattern to fix a circular split.

        The array is rotated until the first element is DIFFERENT from the last element,
        ensuring a clean start/end boundary for subsequent RLE calculation.

        Example: [0, 0, 1, 1, 0] (Start == End, split 0s) -> [0, 1, 1, 0, 0] (Start != End)

        Args:
            sampled_values: A 1D NumPy array of binary values.

        Returns:
            np.ndarray: The cyclically aligned array, or the original if uniform.
        """
        if sampled_values.size <= 1:
            return sampled_values

        array = sampled_values.copy()
        shift_amount = 0

        # We shift until the boundary is clean (Start != End)
        while array[0] == array[-1]:
            # Roll the array by one position
            array = np.roll(array, 1)
            shift_amount += 1

            # Prevent infinite loop if the array is uniform (e.g., all 0s)
            if shift_amount >= array.size:
                # If we wrap around entirely, the array must be uniform and cannot be a valid pattern.
                return sampled_values  # Return original, validation will fail later

        return array

    @staticmethod
    def _validate_rle_structure(
        sampled_values: np.ndarray, num_parts: int, tolerance: int
    ) -> bool:
        """
        STEP 3: Performs Run-Length Encoding (RLE) on the aligned data and
        checks the resulting segment count and length tolerance.

        Args:
            sampled_values: The 1D NumPy array after noise cleaning and alignment.
            num_parts: The expected number of segments (runs).
            tolerance: The maximum allowable deviation in length.

        Returns:
            bool: True if the RLE results in the correct number of segments,
            each within tolerance; False otherwise.
        """
        total_size = sampled_values.size

        # Calculate expected length based on pre-checked divisibility
        expected_len = total_size // num_parts

        # --- RLE Calculation ---
        # Find the indices where the value changes
        change_indices = np.where(np.diff(sampled_values) != 0)[0] + 1

        # If the array is uniform (no changes), return False
        if change_indices.size == 0:
            return False

        # Since the array is ALIGNED (Start != End), the RLE calculation is straightforward.
        # [0] marks the start of the first run; [total_size] marks the end of the last run.
        all_indices = np.concatenate([[0], change_indices, [total_size]])
        run_lengths = np.diff(all_indices)

        # --- Final Validation Checks ---

        # Check 1: We must be left with the exact expected number of segments
        if len(run_lengths) != num_parts:
            return False

        # Check 2: All segments must fall within the expected size range
        lower_bound = expected_len - tolerance
        upper_bound = expected_len + tolerance

        is_valid_run = np.logical_and(
            run_lengths >= lower_bound, run_lengths <= upper_bound
        )

        # All conditions must be true
        return np.all(is_valid_run)

    @staticmethod
    def check_alternating_pattern(
        sampled_values: np.ndarray, num_parts: int = 4, tolerance: int = 1
    ) -> bool:
        """
        The primary pipeline for validating an alternating pattern: Clean -> Align -> Validate.

        Args:
            sampled_values: A 1D NumPy array of strictly binary pixel values (0 or 1).
            num_parts: The expected number of segments (runs) in the final pattern.
            tolerance: The maximum allowable deviation in segment length.

        Returns:
            bool: True if the sequence represents a valid, equally-sized alternating pattern;
            False otherwise.

        Raises:
            ValueError: If the total size of the array is not exactly divisible by num_parts.
        """

        # --- 0. INITIAL VALIDATION ---
        total_size = sampled_values.size
        if total_size % num_parts != 0:
            raise ValueError(
                f"Total sample size ({total_size}) must be exactly divisible by "
                f"the expected number of parts ({num_parts})."
            )

        # 1. CLEAN NOISE: Removes isolated single-pixel errors.
        cleaned_values = BitUtils._clean_noise(sampled_values)

        # 2. ALIGN PATTERN: Fixes circular split by rotation.
        aligned_values = BitUtils._align_pattern(cleaned_values)

        # 3. VALIDATE STRUCTURE: Checks for correct RLE segment count and size tolerance.
        return BitUtils._validate_rle_structure(aligned_values, num_parts, tolerance)
