# Arcane Chess
> Web based and deep-q-learning powered chess game.

Arcane Chess is a modal based on a Deep Reinforcement Learning approach to master the chess game.
You can train your own Reinforced Agent and then deploy-it online and oppose him.

![](header.png)

## Installation


For training the modal :
```sh
git clone https://github.com/raysr/Arcane-Chess
cd Arcane-Chess/app
python3.6 learning/train.py
```

To deploy the local server :
```sh
cd Arcane-Chess/app
python3.6 
```

## Usage example

To play against your trained modal, deploy the local server, then just access http://localhost:5000/

## Development setup

To train your own version of Arcane Chess with specified hyperparameters, you first need some libraries :

```sh
pip3 install tensorflow keras flask numpy python-chess
```
Then you can execute the train script with optional parameters :
```sh
python3 learning/train.py --number_of_games 100000 --winner_reward 1 --loser_malus -1 --epsilon 1
                          --decremental_epsilon 0.0001 --gamma 0.05
```
Each of those ones are explained in the Approach section of this README.

## Approach

## Libraries Used

* python-chess : to generate all the moves and simulate chess engine.
* Flask : forthe bac-kend webserver.
* Keras/TensorFlow : Frameworks used for the Deep Neural Network.
* chessjs/chessboardjs : for the front-end chess engine.

## Meta

Rayan Samy Ramoul – [https://github.com/raysr](https://github.com/raysr)

Distributed under the MIT license. See ``LICENSE`` for more information.


## Contributing

1. Fork it (<https://github.com/raysr/Arcane-Chess/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

