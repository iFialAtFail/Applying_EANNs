/// Author: Samuel Arzt
/// Date: March 2017

#region Includes
public class LongRangeSensor : Sensor
{
    override protected void Start()
    {
        base.Start();
        MAX_DIST = 20f;
    }

}
