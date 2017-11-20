/// Author: Samuel Arzt
/// Date: March 2017

#region Includes
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEditor;
using System.Reflection;
#endregion

/// <summary>
/// Singleton class managing the overall simulation.
/// </summary>
public class GameStateManager : MonoBehaviour
{
    #region Members
    // The camera object, to be referenced in Unity Editor.
    [SerializeField]
    private CameraMovement Camera;

    // The name of the track to be loaded
    [SerializeField]
    public string TrackName;

    public string NewTrackName;

    /// <summary>
    /// The UIController object.
    /// </summary>
    public UIController UIController
    {
        get;
        set;
    }

    public static GameStateManager Instance
    {
        get;
        private set;
    }

    private CarController prevBest, prevSecondBest;
    #endregion

    #region Constructors
    private void Awake()
    {
        if (Instance != null)
        {
            Debug.LogError("Multiple GameStateManagers in the Scene.");
            return;
        }
        Instance = this;

        //Load gui scene
        SceneManager.LoadScene("GUI", LoadSceneMode.Additive);

        //Load track
        SceneManager.LoadScene(TrackName, LoadSceneMode.Additive);
    }

    void Start ()
    {
        TrackManager.Instance.BestCarChanged += OnBestCarChanged;
        EvolutionManager.Instance.StartEvolution();
	}
    #endregion

    #region Methods
    // Callback method for when the best car has changed.
    private void OnBestCarChanged(CarController bestCar)
    {
        if (bestCar == null)
            Camera.SetTarget(null);
        else
            Camera.SetTarget(bestCar.gameObject);
            
        if (UIController != null)
            UIController.SetDisplayTarget(bestCar);
    }
    #endregion

    bool shouldDieFromWallHit = true;
    public void ChangeCarKillBool()
    {
        CarController.SetShouldDieFromWallHit(shouldDieFromWallHit);
        shouldDieFromWallHit = !shouldDieFromWallHit;
    }

    public void ChangeTrackScene()
    {
        if (NewTrackName != TrackName)
        {
            //TrackManager.Instance.RemoveAllCars();
            SceneManager.UnloadSceneAsync(TrackName);
            SceneManager.UnloadSceneAsync("GUI");

            //Load track
            SceneManager.LoadScene(NewTrackName, LoadSceneMode.Additive);
            TrackName = NewTrackName;
            NewTrackName = "";
            
            //Reload gui for current track
            SceneManager.LoadScene("GUI", LoadSceneMode.Additive);

        }
    }


    [InspectorButton("ChangeTrackScene", ButtonWidth = 200f)]
    public bool updateTrackScene;

    [InspectorButton("ChangeCarKillBool", ButtonWidth =200f)]
    public bool changeCarKillBool;
}
