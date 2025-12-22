import Metashape
import csv

# Get the active project chunk
chunk = Metashape.app.document.chunk

# Check if a chunk is loaded
if not chunk:
    raise Exception("No active project chunk found!")

# Define the input file path and name
input_path = Metashape.app.getOpenFileName(
    "Open marker pixel coordinates CSV...", filter="*.csv"
)

if not input_path:
    print("Import cancelled.")
else:
    # A dictionary to hold marker data for batch creation
    markers_to_create = {}

    with open(input_path, "r", newline="") as csv_file:
        reader = csv.reader(csv_file)

        # Skip the header row
        next(reader)

        for row in reader:
            try:
                image_name = row[0]  # Image name with extension
                marker_name = row[1]  # Target ID
                x_coord = float(row[2])  # Target center X coordinate in pixels
                y_coord = float(row[3])  # Target center Y coordinate in pixels

                # Use the image name to find the corresponding camera object
                camera = None
                for cam in chunk.cameras:
                    if cam.label == image_name:
                        camera = cam
                        break

                if not camera:
                    print(
                        f"Warning: image '{image_name}' not found in the project. Skipping marker data for this image."
                    )
                    continue

                # Add the projection data to our dictionary
                if marker_name not in markers_to_create:
                    markers_to_create[marker_name] = {}

                markers_to_create[marker_name][camera] = Metashape.Vector(
                    [x_coord, y_coord]
                )

            except (IndexError, ValueError) as e:
                print(
                    f"Error reading row: {row}. Please check the CSV format. Error: {e}"
                )
                continue

    # Process the gathered data to create and place markers
    if markers_to_create:
        for marker_name, projections in markers_to_create.items():
            # Find or create the marker
            marker = None
            for m in chunk.markers:
                if m.label == marker_name:
                    marker = m
                    break

            if not marker:
                marker = chunk.addMarker()
                marker.label = marker_name

            # Place the projections on the cameras
            for camera, coord in projections.items():
                marker.projections[camera] = Metashape.Marker.Projection(coord)

        print("Import of markers from CSV is complete!")
    else:
        print("No valid marker data found in the CSV file.")
