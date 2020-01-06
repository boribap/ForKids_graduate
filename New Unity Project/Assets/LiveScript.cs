using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using TensorFlow;
using System.Runtime.InteropServices;

public class LiveScript : MonoBehaviour
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

    private float result_rate = 0.0f;
    private int result_label = 0;


    Text textAsset, textAsset1, textAsset2, textAsset3, textAsset4, textAsset5, textAsset6, textAsset7;
    Slider slider1, slider2, slider3, slider4, slider5, slider6, slider7;

    string[] labels = { "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral" };
    float[] percent = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // Start is called before the first frame update
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

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            if (Input.GetMouseButtonDown(0))
            {
                SaveImage();

                //Debug.Log("Show Origin image.");
                //OpenCVInterop.Show();

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

                        var file = "./Assets/cropimg.png";

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

                        for (int i = 0; i < 7; i++)
                        {
                            output = results[0, i];

                            percent[i] = output * 100;

                            if (output >= result_rate)
                            {
                                result_rate = output;
                                result_label = i;
                            }

                        }
                    }
                }

                webcamTexture.Stop();
                byte[] byteArray = File.ReadAllBytes(@"C:\Users\dqf96\Desktop\NewUnityProject\Assets\cropimg.png");
                //create a texture and load byte array to it
                // Texture size does not matter 
                Texture2D sampleTexture = new Texture2D(2, 2);
                // the size of the texture will be replaced by image size
                bool isLoaded = sampleTexture.LoadImage(byteArray);
                // apply this texure as per requirement on image or material
                GameObject image = GameObject.Find("RawImage");
                if (isLoaded)
                {
                    image.GetComponent<RawImage>().texture = sampleTexture;
                }

                // 결과 화면에 표시하기       
                slider1 = GameObject.Find("Canvas1").transform.Find("Slider1").GetComponent<Slider>();
                slider1.value = percent[0];
                textAsset1 = GameObject.Find("Canvas1").transform.Find("result1").GetComponent<Text>();
                textAsset1.text = percent[0] + "%";

                slider2 = GameObject.Find("Canvas1").transform.Find("Slider2").GetComponent<Slider>();
                slider2.value = percent[1];
                textAsset2 = GameObject.Find("Canvas1").transform.Find("result2").GetComponent<Text>();
                textAsset2.text = percent[1] + "%";

                slider3 = GameObject.Find("Canvas1").transform.Find("Slider3").GetComponent<Slider>();
                slider3.value = percent[2];
                textAsset3 = GameObject.Find("Canvas1").transform.Find("result3").GetComponent<Text>();
                textAsset3.text = percent[2] + "%";

                slider4 = GameObject.Find("Canvas1").transform.Find("Slider4").GetComponent<Slider>();
                slider4.value = percent[3];
                textAsset4 = GameObject.Find("Canvas1").transform.Find("result4").GetComponent<Text>();
                textAsset4.text = percent[3] + "%";

                slider5 = GameObject.Find("Canvas1").transform.Find("Slider5").GetComponent<Slider>();
                slider5.value = percent[4];
                textAsset5 = GameObject.Find("Canvas1").transform.Find("result5").GetComponent<Text>();
                textAsset5.text = percent[4] + "%";

                slider6 = GameObject.Find("Canvas1").transform.Find("Slider6").GetComponent<Slider>();
                slider6.value = percent[5];
                textAsset6 = GameObject.Find("Canvas1").transform.Find("result6").GetComponent<Text>();
                textAsset6.text = percent[5] + "%";

                slider7 = GameObject.Find("Canvas1").transform.Find("Slider7").GetComponent<Slider>();
                slider7.value = percent[6];
                textAsset7 = GameObject.Find("Canvas1").transform.Find("result7").GetComponent<Text>();
                textAsset7.text = percent[6] + "%";

                textAsset = GameObject.Find("Canvas1").transform.Find("result").GetComponent<Text>();
                textAsset.text = labels[result_label] + ":" + percent[result_label] + "%";
            }
        }
    }

    // Define the functions which can be called from the .dll.
    internal static class OpenCVInterop
    {
        [DllImport("project4", EntryPoint = "Init")]
        internal static extern int Init();

        [DllImport("project4", EntryPoint = "SetScale")]
        internal static extern int SetScale(int downscale);

        [DllImport("project4", EntryPoint = "Show")]
        internal static extern void Show();

        [DllImport("project4", EntryPoint = "Detect")]
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


