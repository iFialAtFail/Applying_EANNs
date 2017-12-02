using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TimeSliderUIControl : MonoBehaviour {

    public float timeMin = 1f;
    public float timeMax = 5f;

	public void OnSliderChanged(float value)
    {
        Time.timeScale = ((timeMax-timeMin)*value) + timeMin;
    }

    private void Start()
    {
        if (Application.isMobilePlatform)
        {
            timeMax = 4f; // 5x speed is a little much for my Galaxy S7, so I can assume that to fit most mobile requirements, it should be less than 5x.
        }
    }
}
