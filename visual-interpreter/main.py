from transformers import AutoProcessor, AutoModelForCausalLM
import cv2, torch, textwrap

processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
model.eval()

cap = cv2.VideoCapture(0)
caption = ""
frame_skip = 8
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1

    if count % frame_skip == 0:
        inputs = processor(images=frame, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(out[0], skip_special_tokens=True)

    for i, line in enumerate(textwrap.wrap(caption, 40)):
        cv2.putText(frame, line, (20, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("GIT Captioning", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
