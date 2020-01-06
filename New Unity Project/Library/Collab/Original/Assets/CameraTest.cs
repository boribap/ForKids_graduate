using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;
using UnityEngine.UI;
using System.IO;
using TensorFlow;

public class CameraTest : MonoBehaviour
{
    public RawImage rawimage;  //Image for rendering what the camera sees.
    WebCamTexture webcamTexture = null;

    public static List<Vector2> NormalizedFacePositions { get; private set; }
    public static Vector2 CameraResolution;

    /// <summary>
    /// Downscale factor to speed up detection.
    /// </summary>
    private const int DetectionDownScale = 1;

    private bool _ready;
    private int _maxFaceDetectCount = 100;
    private CvCircle[] _faces;

    void Start()
    {
        //Save get the camera devices, in case you have more than 1 camera.
        WebCamDevice[] camDevices = WebCamTexture.devices;

        //Get the used camera name for the WebCamTexture initialization.
        string camName = camDevices[0].name;
        webcamTexture = new WebCamTexture(camName);

        //Render the image in the screen.
        rawimage.texture = webcamTexture;
        rawimage.material.mainTexture = webcamTexture;
        webcamTexture.Play();

        int result = OpenCVInterop.Init();
        
        if (result < 0)
        {
            if (result == -1)
            {
                Debug.LogWarningFormat("[{0}] Failed to find cascades definition.", GetType());
            }
            else if (result == -3)
            {
                Debug.LogWarningFormat("[{0}] Failed to open image.", GetType());
            }

            return;
        }

        //CameraResolution = new Vector2(camWidth, camHeight);
        _faces = new CvCircle[_maxFaceDetectCount];
        NormalizedFacePositions = new List<Vector2>();
        OpenCVInterop.SetScale(DetectionDownScale);
        _ready = true;
    }

    void Update()
    {
        //This is to take the picture, save it and stop capturing the camera image.
        if (Input.GetMouseButtonDown(0))
        {
            SaveImage();
            //webcamTexture.Stop();

            Debug.Log("Show Origin image.");
            OpenCVInterop.Show();

            if (!_ready)
                return;

            int detectedFaceCount = 0;
            unsafe
            {
                fixed (CvCircle* outFaces = _faces)
                {
                    Debug.Log("Detect Start.");
                    //outFaces
                    OpenCVInterop.Detect(outFaces, _maxFaceDetectCount, ref detectedFaceCount);

                    //
                    //Tensorflow code
                    //
                    string PATH = "cropimg.png";    //이미지 위치를 저장하는 변수
                    var testImage = Resources.Load(PATH, typeof(Texture2D)) as Image;  //이미지 로드

                    var file = "C:/Users/bsww201/Desktop/New Unity Project/New Unity Project/Assets/cropimg.png";

                    //Tensor 불러오는 소스
                    TFSession.Runner runner;

                    TextAsset graphModel = Resources.Load("tf_model_191203_05") as TextAsset;
                    var graph = new TFGraph();
                    //graph.Import(new TFBuffer(graphModel.bytes));
                    graph.Import(graphModel.bytes);
                    TFSession session = new TFSession(graph);

                    Debug.Log("loaded freezed graph");

                    // Input , output 설정 
                    //int inputSize = 48;
                    //Texture2D img_input = testImage;
                    //TFTensor input_tensor = TransformInput(img_input.GetPixels32(), inputSize, inputSize);
                    //SetScreen(testImage.width, testImage.height, rawimage, testImage);

                    var tensor = CreateTensorFromImageData(file);

                    runner = session.GetRunner();
                    runner.AddInput(graph["input_1"][0], tensor);
                    runner.Fetch(graph["predictions/Softmax"][0]);

                    Debug.Log("fetch finish");

                    // 실행
                    float[,] results = runner.Run()[0].GetValue() as float[,];

                    Debug.Log("run");

                    float output = 0.0f;
                    string[] labels = { "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral" };
                    for (int i=0; i<7; i++)
                    {
                        output = results[0, i];

                        Debug.Log(labels[i] + " : " + output);

                    }
                }
            }
        }

    }

    // Define the functions which can be called from the .dll.
    internal static class OpenCVInterop
    {
        [DllImport("project1202", EntryPoint = "Init")]
        internal static extern int Init();

        [DllImport("project1202", EntryPoint = "SetScale")]
        internal static extern int SetScale(int downscale);

        [DllImport("project1202", EntryPoint = "Show")]
        internal static extern void Show();

        [DllImport("project1202", EntryPoint = "Detect")]
        internal unsafe static extern void Detect(CvCircle* outFaces, int maxOutFacesCount, ref int outDetectedFacesCount);
    }


    // Define the structure to be sequential and with the correct byte size (3 ints = 4 bytes * 3 = 12 bytes)
    [StructLayout(LayoutKind.Sequential, Size = 12)]
    public struct CvCircle
    {
        public int X, Y, Radius;
    }

    static TFTensor CreateTensorFromImageData(string file)
    {

        var contents = File.ReadAllBytes(file);
        var tensor = TFTensor.CreateString(contents);

        TFGraph graph;
        TFOutput input, output;

        // Construct a graph to normalize the image
        ConstructGraphToNormalizeImage(out graph, out input, out output);

        // Execute that graph to normalize this one image
        using (var session = new TFSession(graph))
        {
            var normalized = session.Run(
                inputs: new[] { input },
                inputValues: new[] { tensor },
                outputs: new[] { output });

            return normalized[0];
        }
    }

    static void ConstructGraphToNormalizeImage(out TFGraph graph, out TFOutput input, out TFOutput output)
    {
        const int W = 48;
        const int H = 48;
        const float Mean = 0.5f;
        const float Scale = 255.0f;
        const int channels = 1;
        const float divide = 2.0f;

        graph = new TFGraph();
        input = graph.Placeholder(TFDataType.String);

        output = graph.Mul(
            x: graph.Sub(
                x: graph.Div(
                  x: graph.ResizeBilinear(
                    images: graph.ExpandDims(
                        input: graph.Cast(
                            graph.DecodePng(contents: input, channels: channels), DstT: TFDataType.Float),
                        dim: graph.Const(0, "make_batch")),
                    size: graph.Const(new int[] { W, H }, "size")),
                  y: graph.Const(Scale, "scale")
                ),
                y: graph.Const(Mean, "mean")
            ),
            y: graph.Const(divide, "divide")
            );
    }

    void SaveImage()
    {
        //Create a Texture2D with the size of the rendered image on the screen.
        Texture2D texture = new Texture2D(rawimage.texture.width, rawimage.texture.height, TextureFormat.ARGB32, false);

        //Save the image to the Texture2D
        texture.SetPixels(webcamTexture.GetPixels());
        texture.Apply();

        //Encode it as a PNG.
        byte[] bytes = texture.EncodeToPNG();

        //Save it in a file.
        File.WriteAllBytes(Application.dataPath + "/testimg.png", bytes);
    }
}