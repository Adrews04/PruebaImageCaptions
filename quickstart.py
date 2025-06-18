import os
import dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

dotenv.load_dotenv()

try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment cariables 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this script")
    exit()

#Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)


visual_features =[
    VisualFeatures.TAGS,
    VisualFeatures.OBJECTS,
    VisualFeatures.CAPTION,
    VisualFeatures.DENSE_CAPTIONS,
    VisualFeatures.READ,
    VisualFeatures.SMART_CROPS,
    VisualFeatures.PEOPLE,
]

#LOCAL IMAGE-----------------------------------------------------------------
with open("photo.jpg", "rb") as f:
    image_data = f.read()

# Get caption for the image. Synchronously.
result = client._analyze_from_image_data(
    image_data=image_data,
    visual_features=visual_features,
    gender_neutral_caption=True,
    language="en",
)

""" #WEB IMAGE-----------------------------------------------------------------
image_url="https://images.unsplash.com/photo-1501386761578-eac5c94b800a?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Y29uY2VydCUyMHBlb3BsZXxlbnwwfHwwfHx8MA%3D%3D",

# Get caption for the image. Synchronously.
result = client._analyze_from_url(
    image_url=image_url,
    visual_features=visual_features,
    gender_neutral_caption=True,
    language="en",
) """


print("Image analysis results:")

# Print all analysis results to the console
print("Image analysis results:")

if result.caption is not None:
    print(" Caption:")
    print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

if result.dense_captions is not None:
    print(" Dense Captions:")
    for caption in result.dense_captions.list:
        print(f"   '{caption.text}', {caption.bounding_box}, Confidence: {caption.confidence:.4f}")

if result.read is not None:
    print(" Read:")
    for line in result.read.blocks[0].lines:
        print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

if result.tags is not None:
    print(" Tags:")
    for tag in result.tags.list:
        print(f"   '{tag.name}', Confidence {tag.confidence:.4f}")

if result.objects is not None:
    print(" Objects:")
    for object in result.objects.list:
        print(f"   '{object.tags[0].name}', {object.bounding_box}, Confidence: {object.tags[0].confidence:.4f}")

if result.people is not None:
    print(" People:")
    for person in result.people.list:
        print(f"   {person.bounding_box}, Confidence {person.confidence:.4f}")

if result.smart_crops is not None:
    print(" Smart Cropping:")
    for smart_crop in result.smart_crops.list:
        print(f"   Aspect ratio {smart_crop.aspect_ratio}: Smart crop {smart_crop.bounding_box}")


print(f" Model version: {result.model_version}")