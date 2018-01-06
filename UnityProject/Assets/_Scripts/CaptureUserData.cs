using System.Collections;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Text;

public class CaptureUserData : MonoBehaviour
{

    public static List<double[]> allData = new List<double[]>();

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public static void addDataToLog(double[] theData)
    {
        allData.Add(theData);
    }

    public static void WriteToFile(string filePath)
    {
        StringBuilder sb = new StringBuilder();
        foreach (var doubleArray in allData)
        {
            foreach (double value in doubleArray)
            {
                sb.AppendFormat("{0},", value);
            }
            sb.AppendLine();
        }
        //string json = JsonUtility.ToJson(allData, true);
        File.WriteAllText(filePath, sb.ToString());

        //Debug.LogFormat("WriteToFile({0}) -- data:\n{1}", filePath, json);
    }

    [MenuItem("Misc/SaveUserData")]
    private static void OnDetroy()
    {
        if (allData.Count > 0)
        {
            WriteToFile(@"C:\Users\USER\Documents\drivingData.txt");
        }
    }
}
