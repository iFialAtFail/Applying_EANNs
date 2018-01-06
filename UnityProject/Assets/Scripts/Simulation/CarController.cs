/// Author: Samuel Arzt
/// Date: March 2017

#region Includes
using UnityEngine;
using System.Collections.Generic;
#endregion

/// <summary>
/// Class representing a controlling container for a 2D physical simulation
/// of a car with 5 front facing sensors, detecting the distance to obstacles.
/// </summary>
public class CarController : MonoBehaviour
{
    #region Members
    #region IDGenerator
    // Used for unique ID generation
    private static int idGenerator = 0;
    /// <summary>
    /// Returns the next unique id in the sequence.
    /// </summary>
    private static int NextID
    {
        get { return idGenerator++; }
    }
    #endregion

    // Maximum delay in seconds between the collection of two checkpoints until this car dies.
    private const float MAX_CHECKPOINT_DELAY = 7;
    private long count = 0;
    /// <summary>
    /// The underlying AI agent of this car.
    /// </summary>
    public Agent Agent
    {
        get;
        set;
    }

    public float CurrentCompletionReward
    {
        get { return Agent.Genotype.Evaluation; }
        set { Agent.Genotype.Evaluation = value; }
    }

    /// <summary>
    /// Whether this car is controllable by user input (keyboard).
    /// </summary>
    public bool UseUserInput = false;

    /// <summary>
    /// The movement component of this car.
    /// </summary>
    public CarMovement Movement
    {
        get;
        private set;
    }

    /// <summary>
    /// The current inputs for controlling the CarMovement component.
    /// </summary>
    public double[] CurrentControlInputs
    {
        get { return Movement.CurrentInputs; }
    }

    /// <summary>
    /// The cached SpriteRenderer of this car.
    /// </summary>
    public SpriteRenderer SpriteRenderer
    {
        get;
        private set;
    }

    private Sensor[] sensors;
    private float timeSinceLastCheckpoint;
    #endregion

    #region Constructors
    void Awake()
    {
        //Cache components
        Movement = GetComponent<CarMovement>();
        SpriteRenderer = GetComponent<SpriteRenderer>();
        sensors = GetComponentsInChildren<Sensor>();
    }
    void Start()
    {
        Movement.HitWall += Die;

        //Set name to be unique
        this.name = "Car (" + NextID + ")";
    }
    #endregion

    #region Methods
    /// <summary>
    /// Restarts this car, making it movable again.
    /// </summary>
    public void Restart()
    {
        Movement.enabled = true;
        timeSinceLastCheckpoint = 0;

        foreach (Sensor s in sensors)
            s.Show();

        Agent.Reset();
        this.enabled = true;
    }

    // Unity method for normal update
    void Update()
    {
        timeSinceLastCheckpoint += Time.deltaTime;
    }

    // Unity method for physics update
    void FixedUpdate()
    {
        //Get control inputs from Agent
        if (!UseUserInput)
        {
            //Get readings from sensors
            double[] sensorOutput = new double[sensors.Length + 1];
            for (int i = 0; i < sensors.Length-1; i++)
                sensorOutput[i] = sensors[i].Output;
            sensorOutput[sensorOutput.Length - 1] = Movement.Velocity;//Might need to normallize this data.
            double[] controlInputs = Agent.FNN.ProcessInputs(sensorOutput);
            Movement.SetInputs(controlInputs);
        }
        else
        {
            if (++count % 20 == 0)
            {
                List<double> data = new List<double>();
                //Get readings from sensors
                double[] sensorOutput = new double[sensors.Length + 1];
                for (int i = 0; i < sensors.Length - 1; i++)
                    data.Add(sensors[i].Output);
                foreach (var item in CurrentControlInputs)
                {
                    data.Add(item);
                }
                CaptureUserData.addDataToLog(data.ToArray());
                if (count > 10000) count = 0;
            }
        }

        if (timeSinceLastCheckpoint > MAX_CHECKPOINT_DELAY)
        {
            Die();
        }
    }

    // Makes this car die (making it unmovable and stops the Agent from calculating the controls for the car).
    private void Die()
    {
        this.enabled = false;
        Movement.Stop();
        Movement.enabled = false;

        foreach (Sensor s in sensors)
            s.Hide();

        Agent.Kill();
    }

    public void CheckpointCaptured()
    {
        timeSinceLastCheckpoint = 0;
    }
    #endregion
}
