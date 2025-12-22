# --------------------------------------------
# StringUtils class
# --------------------------------------------
class StringUtils:
    @staticmethod
    def replace_char_at_index(target_string: str, source_string: str, index: int) -> str:
        """
        Replaces a single character in the target_string at a given index 
        with the character from the source_string at the same index.

        Since strings are immutable, this function returns a new string.

        Args:
            target_string (str): The string to be modified (the 'base').
            source_string (str): The string providing the replacement character.
            index (int): The position (0-based) of the character to replace 
                        and the position of the character to use as the replacement.

        Returns:
            str: A new string with the character at the specified index replaced.

        Raises:
            IndexError: If the index is out of bounds for either the target 
                        or the source string.
        """
        # ------------------------------------------------------------------
        # --- Input Validation ---
        
        # Check if the index is valid for both strings
        if not (0 <= index < len(target_string) and 0 <= index < len(source_string)):
            raise IndexError(
                f"Index {index} is out of bounds for string lengths "
                f"({len(target_string)}, {len(source_string)})"
            )

        # ------------------------------------------------------------------
        # --- String Replacement using Slicing ---

        # 1. Get the character to be inserted
        new_char = source_string[index]

        # 2. Build the new string using concatenation:
        #    a. target_string[:index]     -> Part of target string BEFORE the index
        #    b. new_char                  -> The replacement character
        #    c. target_string[index + 1:] -> Part of target string AFTER the index
        
        new_string = target_string[:index] + new_char + target_string[index + 1:]
        
        return new_string