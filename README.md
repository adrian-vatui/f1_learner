# Car Racing DDPG agent

## By: Chiriac Cătălin, Vătui Adrian

### Problem description

We used the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) environment from [Gym](https://gym.openai.com/) to
teach a racing car to drive itself in any randomly generated track, aiming to obtain a score as high as possible. The
way the score is calculated is as follows:
The reward is -0.1 for every frame and + 1000/N for every track tile visited, where N is the total number of tiles in
the track (basically 1 for every track). There are no restrictions that force the car to not drive on grass and there
are also no penalties for this action. The only way the game forcefully ends is if the car drives out of the map's
bounds, in which case the score reward is -100, or when 1000 frames pass. At the bottom of the screen, there are also
some indicators - these are, in order: the score, the true speed, 4 ABS sensors, steering wheel position, and gyroscope.
Each state consists of a 96x96-pixel image. It's considered that the car has finished learning if it's able to obtain a
score of 900 points or more.

### Development

#### Image precessing

One of the parts which we probably modified the most is the image pre-processing. Before we even started working on the
project we researched and found many ideas that seemed good, and we also came with our new ideas such as:

1) image cropping (remove the 12 pixels bar at the bottom and 6 pixels from the left and right margins so that the new
   image is 84x84)
2) converting to grayscale ( dimension of 84x84x1 instead of 84x84x3)
3) normalize the values between 0 and 1, dividing them by 255
4) unifying the grass color
5) unifying the road color
6) modifying and unifying the car color the have better contrast with the grayscaled image
7) separating the bottom bar from the rest of the image and processing them separately

We thought that by combining these ideas we would probably get a score good (or close) enough to pass the learning
criteria even with a basic network. As we started experimenting, trying different combinations, and observing the car's
behavior, we concluded that some ideas were doing more harm than good.

1) Cropping the image made it harder for the neural network to realize the speed at which the car was moving and also
   how good the score was
2) Unlike the grass color unification, unifying the road color made it harder to realize if the car was moving backward
   or on the grass (since the score obtained was identical). Without it, as the car passed a tile, its color would've
   changed, and the agent would've known the car had already passed through that area.
3) We needed to unify the colors RED and BLACk in a specific area for the car, but as the car drifted, black marks
   appeared on the road, which is why after modifying the color, the shape of it would sometimes be weird.
4) We got similar results, but the process was more costly.

In the end, the best combination included enlarged speed bar, grass color unification, and converting to grayscale.

#### Different agent types

We started out by trying a simple Q-learning solution that only used one network to approximate the Q-value and decide
between 5 discrete actions (do nothing, left, right, accelerate, break). The agent however did not perform that well,
not even getting a positive score.

We then moved on to another approach - Deep Deterministic Policy Gradient. This solution uses 4 networks (actor, critic
and one target network for each) in order to learn off-policy and output actions in a continuous search space. We
learned a lot about the algorithm and took inspiration
from [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
and [here](https://keras.io/examples/rl/ddpg_pendulum/).

### Results

The best configuration we obtained has an average score of 800 points. It tends to drive on the side of the road (to
have more space to turn in curves) and to drive at high speeds to get more points quickly. Sometimes it gets a lower
score because the car starts going backwards on the track after spinning out, since it doesn't know what the right way
forward is. It tends to finish the course before the time limit (after which the score doesn't increase anymore). You
can clearly see it has learned to break before sharp turns.

The networks in the bestSolution folder, after training for 600 episodes and with an average score around 800-850:
![bestSolution (1)](https://user-images.githubusercontent.com/75117511/146388889-1dd3cebe-840c-4759-a3b1-93bda303c161.gif)

After more training, we can see that the agent learns to favour speed over control and starts cutting corners and
driving alarmingly fast, often finishing more than 1 and a third complete circuits. This leads to lower scores, partly
because of the "missed" tiles in curves, partly because of the increased frequency of spin-outs and drifting. This
particular model was even trained with a further reduction of speed of 1.5 instead of 1.25, but it just lowered the
acceleration, not the maximum speed.

The networks in the bestConfig2 folder, after training for 2100 episodes and with an average score of 740:
![bestConfig2 (1)](https://user-images.githubusercontent.com/75117511/146389028-b43ebc63-e76d-463f-b007-09ece1aa6812.gif)

### Network architecture

#### Basic agent

For the basic agent, we used 2 convolutional layers with max pooling between them, followed by a 64-perceptron dense
layer and an output layer of 5 neurons, each of them representing the one of the 5 discrete actions. To decide on an
action, we simply picked the one with the largest value.
![basic agent architecture](basic_agent.png)

#### DDPG actor

The architecture is very similar to the one we used before, the only exception being that we added a Gaussian Noise
layer, and the output layer has only 2 neurons since we only need 2 values (steering and acceleration/brake).
![DDPG actor architecture](actor.png)

#### DDPG critic

The critic also uses the same structure as the other networks, but its first dense layer also takes the output of the
actor as input. Its output layer has only one neuron, since it just needs to approximate `Q(state, action)`.
![DDPG critic architecture](critic.png)

### Implementation details

Some interesting "hacks" we used:

* The actor uses tanh as the activation function for its final layer, since it outputs values in the [-1, 1] range,
  which is exactly what is needed for the action space
* Even though an action needs 3 values, the actor only outputs 2, since braking and accelerating at the same time is
  never needed
* The actor is polled every 2 frames - this is to diversify the memory buffer by skipping some frames; polling too
  rarely would make the movements of the car very chaotic, driving from the left to the right margin of the road
* Actions are divided by 4 to make the control easier and the speed divided by a further 1.25 to try and make it drive
  more slowly.
* The actor's output has some Gaussian noise added. After obtaining a score of over 800 (very good), we remove the noise
  in order to force the network to exploit its current configuration, but if the score drops, the noise is added back to
  encourage exploration.