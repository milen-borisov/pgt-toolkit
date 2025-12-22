
# --------------------------------------------
# ListUtils class
# --------------------------------------------
class ListUtils:

    @staticmethod
    def generate_alternating_sequence(start_value: int, step_value: int, bound_down_value: int, bound_up_value: int) -> list[int]:
        """
        Generates a list of values by alternating between subtracting and adding the 
        step_value to the start_value, while staying within the specified bounds.
        
        For example: generate_alternating_sequence(143, 10, 100, 170) -> [133, 153, 123, 163, 113, 103]

        Args:
            start_value: The initial value to begin the sequence from.
            step_value: The magnitude of the step for each iteration.
            bound_down_value: The lower (exclusive) bound.
            bound_up_value: The upper (exclusive) bound.

        Returns:
            A list of generated values in the alternating pattern.
        """
        
        # Initialize the list to store the results
        result_list = []
        
        # Initialize the current step multiplier to 1 (for the first step: start_value - step_value)
        # The actual step will be step_value * multiplier. 
        # This multiplier alternates between 1, 2, 3, ...
        step_multiplier = 1
        
        # Loop indefinitely until both the down and up sequences exceed their respective bounds
        while True:
            # --- Down Step (start_value - (step_value * step_multiplier)) ---
            
            # Calculate the potential value for the "down" step
            down_value = start_value - (step_value * step_multiplier)
            
            # Check if the down_value is within the lower bound
            # The example output suggests that the bounds are exclusive: (143, 10, 100, 170) -> 103 is included, so 100 is exclusive.
            if down_value > bound_down_value:
                # If within bounds, add it to the result list
                result_list.append(down_value)
                # Flag that we successfully added a value
                down_added = True
            else:
                # If it's out of bounds, no more down values can be added
                down_added = False
                
            # --- Up Step (start_value + (step_value * step_multiplier)) ---
            
            # Calculate the potential value for the "up" step
            up_value = start_value + (step_value * step_multiplier)
            
            # Check if the up_value is within the upper bound
            if up_value < bound_up_value:
                # If within bounds, add it to the result list
                result_list.append(up_value)
                # Flag that we successfully added a value
                up_added = True
            else:
                # If it's out of bounds, no more up values can be added
                up_added = False
            
            # --- Termination Condition ---
            
            # If neither a down value nor an up value was successfully added in this iteration, 
            # it means both sequences have hit their boundaries, so we stop the loop.
            if not down_added and not up_added:
                break
                
            # Prepare for the next iteration (increase the step multiplier)
            step_multiplier += 1

        return result_list