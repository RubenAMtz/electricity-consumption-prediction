# Electricity consumption prediction

The mining industry is one of the greatest consumers of natural resources to power their machines, the prediction of process variables 
could mean saving massive amount of these resources.

# Overview

The mining process consists of extracting rocks from the ground and ultimately processing this rocks to extract rich material. 
Usually rocks have to go through different stages of size reduction reaching eventually the concentration plant where mills (usually 
SAG mills and later ball mills) finally fine tune the size of the rocks in the order of the micro milimeters. As grinding mills are the
last stage of size reduction process control engineers have to make sure that the size of the rocks coming out of them meet a given
quality requirement, those rocks that do not pass the quality test have to be recirculated to the mill creating a feeback loop.

![Mills](https://github.com/RubenAMtz/electricity-consumption-prediction/blob/master/mining.jpg)

Because the feedback loop can get out of control, bringing the mills to a full stop and ultimately stopping the whole process, 
process control engineers and control room operators have to become sharp process supervisors looking for those signals that might
indicate the trend towards an undesired state.

For this challenge microphones have been setup around the mill station to record the audio produced by the movement of the internal
material:

We are asked to predict the power consumption of the mill using audio data.

# Data structure

Target range: 5200 - 5835 kWh

- 126 minutes of audio (tdms files)
- Process variable sampled every minute, hence, 126 samples

| Commit       | Score (mae) | Date     |
| ------------ | ----------- | -------- |
| First        | 157.12      | 10/03/19 |
