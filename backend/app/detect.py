print("Argus is starting....")

import cv2
from insightface.app import FaceAnalysis
import numpy as np
import os

print("InsightFace is imported")

EMBEDDINGS_DIR = "../data/embeddings"


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def load_known_faces():
    known_faces = {}

    if not os.path.exists(EMBEDDINGS_DIR):
        return known_faces

    for person_name in os.listdir(EMBEDDINGS_DIR):

        person_dir = os.path.join(
            EMBEDDINGS_DIR,
            person_name
        )

        if not os.path.isdir(person_dir):
            continue

        # Store multiple embeddings for each person
        embeddings = []

        for filename in os.listdir(person_dir):

            if filename.endswith(".npy"):

                embedding = np.load(
                    os.path.join(person_dir, filename)
                )

                embeddings.append(embedding)

        if embeddings:
            known_faces[person_name] = embeddings

    return known_faces


def main():

    # Initialize InsightFace
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    # Load known faces
    known_faces = load_known_faces()
    current_person = None

    frame_count = 0
    recognition_interval = 10
    tracks = {}
    next_track_id = 0
    # Last recognized result
    # name = "Unknown"
    # similarity = 0.0

    print(f"Loaded {len(known_faces)} known people")

    for name in known_faces:
        print(f" {name}")

    # Initialize OpenCV
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Camera is not available")

    print("Argus is active")
    print("Press 'q' to stop")
    print("Press 'e' to save a face")
    print("Press 'n' to save a new person's name")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read the frame")
            break

        # Increment frame count
        frame_count += 1

        # Detect faces
        faces = app.get(frame)

        current_faces = []

        for face in faces:
            x1 , y1, x2 , y2 = map(int,face.bbox)
            center = get_center(
                (x1 ,y1 , x2,y2)
            )
            current_faces.append({
                "face" : face,
                "bbox" : (x1,y1,x2,y2),
                "center" : center
            })

            # Match current faces with exsisting faces
            used_tracks = set()

            for current in current_faces:
                face = current["face"]
                bbox = current["bbox"]
                center = current["center"]
                x1,y1,x2,y2 = bbox

                best_track_id =None
                best_distance = float("inf")     

                for track_id,track in tracks.items():
                    if track_id in used_tracks:
                        continue
                    old_center = track["center"]
                    distance = np.linalg.norm(
                        np.arry(center) - np.array(old_center)
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
                if best_track_id is None or best_distance > 100:
                    best_track_id = next_track_id
                    next_track_id += 1

                    tracks[best_track_id] = {
                        "center" : center,
                        "name" : "unknown",
                        "similarity" : 0.0
                    }
                used_tracks.add(best_track_id)
                track = tracks[best_track_id]

                # Update the track position
                track["center"] = center
                # Recognise every 10 frames
                if frame_count % recognition_interval == 0:
                    live_embedding = face.embedding
                    best_name = "unknown"
                    best_similarity = -1

                    for person_name, known_embeddings in known_faces.items():
                        for known_embedding in known_embeddings:
                            similarity = cosine_similarity(
                                live_embedding,
                                known_embeddings
                            )

                            if similarity > best_similarity:
                                 best_similarity = similarity
                                 best_name = name

                    threshold = 0.5

                    if best_similarity >= threshold:
                        track["name"] = best_name
                    else:
                        track["name"] = "Unknown"
                    track["similarity"] = best_similarity

                    # Make the Bounding Boxes

                    cv2.rectangle(
                        frame,
                        (x1,y1),
                        (x2,y2),
                        (0,255,0),
                        2
                    )
                    label = (
                        f"ID {best_track_id} |"
                        f"{tracks['name']} | "
                        f"{tracks['similarity']: .2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10 , 20)),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0,255,0)
                        2
                    )
                    
        # Show frame
        cv2.imshow("Argus", frame)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):

            current_person = input(
                "Enter the new name: "
            ).strip().lower()

            if current_person:
                print(
                    f"Selected person: {current_person}"
                )
            else:
                current_person = None
                print("Invalid name")

        if key == ord('e'):

            if len(faces) == 0:
                print("No Faces found")

            else:

                if current_person is None:
                    print(
                        "Select a person first using 'n' key"
                    )
                    continue

                face = faces[0]
                embedding = face.embedding

                person_name = current_person

                person_dir = os.path.join(
                    EMBEDDINGS_DIR,
                    person_name
                )

                os.makedirs(
                    person_dir,
                    exist_ok=True
                )

                existing_files = [
                    f for f in os.listdir(person_dir)
                    if f.endswith(".npy")
                ]

                image_number = len(existing_files) + 1

                file_path = os.path.join(
                    person_dir,
                    f"{image_number}.npy"
                )

                np.save(
                    file_path,
                    embedding
                )

                def get_center(bbox):
                    x1, y1 ,x2, y2 = bbox
                    return (
                        (x1 , x2) //2
                        (y1 , y2) //2
                    )

                known_faces = load_known_faces()

                print(
                    f"Loaded {len(known_faces)} known people"
                )

                print(
                    f"\nSaving embedding for {person_name}"
                )
                print(f"Saved: {file_path}")
                print("Shape:", embedding.shape)

        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()