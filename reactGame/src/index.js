import React from 'react';
import ReactDOM from 'react-dom';
import Connect4 from './components/Connect4';
import Minesweeper from './components/Minesweeper';
import TicTacToe from './components/TicTacToe';
import registerServiceWorker from './registerServiceWorker';

class App extends React.Component{
    constructor(...args) {
    	super(...args);
    	this.state = {
    		game: 'connect4'
    	}
    }

	render() {
		const { game } = this.state;
        return (
            <div>
                <div>
                    <span
                      style={{ marginRight: '30px', cursor: 'pointer', fontWeight: game === 'connect4' ? 'bold' : null }}
                      onClick={() => this.setState({ game: 'connect4' })}
                    >
                        Connect4
                    </span>
                    <span
                      style={{ marginRight: '30px', cursor: 'pointer', fontWeight: game === 'tictactoe' ? 'bold' : null  }}
                      onClick={() => this.setState({ game: 'tictactoe' })}
                    >
                        TicTacToe
                    </span>
                    <span
                      style={{ marginRight: '30px', cursor: 'pointer', fontWeight: game === 'minesweeper' ? 'bold' : null  }}
                      onClick={() => this.setState({ game: 'minesweeper' })}
                    >
                        Minesweeper
                    </span>
                </div>
                {game === 'connect4' ? (
                    <div>
                        <h1>Connect4</h1>
                        <Connect4 height={8} width={10} connect={4} />
                    </div>
                ) : null}
                {game === 'tictactoe' ? (
                    <div>
                        <h1>Tic Tac Toe</h1>
                        <TicTacToe size={3} />
                    </div>
                ) : null}
                {game === 'minesweeper' ? (
                    <div>
                        <h1>Minesweeper</h1>
                        <Minesweeper height={9} width={9} minesNum={10} />
                    </div>
                ) : null}
            </div>
        );		
	}
}

ReactDOM.render(<App />, document.getElementById('root'));
registerServiceWorker();
