# f1_learner

<p>Observation! The best configuration obtained untill now, has an average score of 800 points. It tends to drive on the side of the road(to take the curves easier) and to accelerate a lot. Some times it gets a lower result because the car starts going backwards on the track after taking a curve at high speed. It tends to finish the course before the time limit (after which the score doesn't increase)</p>

<p>The main specifications: SKIP_FRAMES = 2, actions divided by 4 to make the control easier and the speed divided by a further 1.25 to try and make it drive slower. Also it tries to exploit a configuration with the score above 800 for the next 10 episoded by removing the noise temporarly.</p>
<p>Training episodes: 600</p>

The networks in the bestSolutin folder:

![bestSolution (1)](https://user-images.githubusercontent.com/75117511/146388889-1dd3cebe-840c-4759-a3b1-93bda303c161.gif)



<p>The next configuration has an average score of 740. This configuration is bassically a driving maniac which cuts all the corners/curves and speeds like crazy, usually finishing the whole course and another 3rd of it.</p>
<p>The main specifications: Similar to the one above but the speed was divided by a furget 1.5 instead of 1.25. Also, before saving the best solution it looks if the average score of the last 10 episodes if at least 75% of the current best score.</p>
<p>Training episodes: 2100</p>

The networks in the bestConfig2 folder:

![bestConfig2 (1)](https://user-images.githubusercontent.com/75117511/146389028-b43ebc63-e76d-463f-b007-09ece1aa6812.gif)
