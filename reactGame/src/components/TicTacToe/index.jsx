import React from 'react';
import propTypes from 'prop-types';

import * as styles from './index.css';

const marks = ['╳', '◯']

export class TicTacToe extends React.Component {
    static propTypes = {
    	size: propTypes.number.isRequired,
    }

    constructor(...args) {
    	super(...args);
    	const { size } = this.props;
    	this.state = {
            boxes: Array(size).fill(0).map(r => Array(size).fill(null)),
            steps: 0,
            turn: 0,
            winner: null,
    	};
    }

    checkWinner = (rowIdx, boxIdx, turn) => {
        const { size } = this.props;
        const { boxes } = this.state;
        let up = 0;
        let down = 0;
        let right = 0;
        let left = 0;
        let leftTop = 0;
        let rightBottom = 0;
        let rightTop = 0;
        let leftBottom = 0;

        // check vertical
        let counter = 1;
        while (counter < size && (rowIdx - counter) >= 0) {
            if (boxes[rowIdx - counter][boxIdx] === turn) {
                up += 1;
                counter += 1;
            } else {
                break;
            }
        }
        if (up === (size - 1)) {
            return turn;
        }

        counter = 1;
        while (counter < size && (rowIdx + counter) < size) {
            if (boxes[rowIdx + counter][boxIdx] === turn) {
                down += 1;
                counter += 1;
            } else {
                break;
            }
        }
        if (up + down === (size - 1)) {
            return turn;
        }

        // check right
        counter = 1;
        while (counter < size && (boxIdx + counter) < size) {
            if (boxes[rowIdx][boxIdx + counter] === turn) {
                right += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (right === (size - 1)) {
            return turn;
        }
        
        // check left
        counter = 1;
        while (counter < size && (boxIdx - counter) >= 0) {
            if (boxes[rowIdx][boxIdx - counter] === turn) {
                left += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (right + left === (size - 1)) {
            return turn;
        }        

        // check left diagonal
        counter = 1;
        while (counter < size && (boxIdx - counter) >= 0 && (rowIdx - counter) >= 0) {
            if (boxes[rowIdx - counter][boxIdx - counter] === turn) {
                leftTop += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        counter = 1;
        
        while (counter < size && (boxIdx + counter) < size && (rowIdx + counter) < size) {
            if (boxes[rowIdx + counter][boxIdx + counter] === turn) {
                rightBottom += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (leftTop + rightBottom >= (size - 1)) {
            return turn
        }

        // check right diagonal
        counter = 1;
        while (counter < size && (boxIdx + counter) < size && (rowIdx - counter) >= 0) {
            if (boxes[rowIdx - counter][boxIdx + counter] === turn) {
                rightTop += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        counter = 1;
        
        while (counter < size && (boxIdx - counter) >= 0 && (rowIdx + counter) < size) {
            if (boxes[rowIdx + counter][boxIdx - counter] === turn) {
                leftBottom += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (rightTop + leftBottom >= (size - 1)) {
            return turn
        }

        return null;
    }

    handleClick = (rowIdx, boxIdx) => {
        const { boxes, turn, winner } = this.state;
        if (boxes[rowIdx][boxIdx] === null && winner === null) {
        	boxes[rowIdx][boxIdx] = turn;
        	this.setState((prevState) => {
                return {
                	boxes,
                	steps: prevState.steps + 1,
                	turn: prevState.turn === 0 ? 1 : 0,
                	winner: this.checkWinner(rowIdx, boxIdx, turn),
                }
        	})
        }
    }

    handleRestart = () => {
        const { size } = this.props;
        this.setState({
            boxes: Array(size).fill(0).map(r => Array(size).fill(null)),
            steps: 0,
            turn: 0,
            winner: null,
        })
    }

    render() {
    	const { boxes, steps, winner } = this.state;
    	const { size } = this.props;
    	return (
    		<div>
    		    {boxes.map((row, rowIdx) => 
    		    	<div className={styles.row} key={`row${rowIdx}`}>
    		    	    {row.map((box, boxIdx) =>
    		    	    	<div 
    		    	    	  className={styles.box}
    		    	    	  key={`box${boxIdx}`}
    		    	    	  onClick={() => this.handleClick(rowIdx, boxIdx)}
    		    	    	>
    		    	    	    <span className={styles.font}>{marks[box]}</span>
    		    	    	</div>
    		    	    )}
    		    	</div>
    		    )}
                <button type="button" onClick={this.handleRestart}>
                    {steps === size * size || winner !== null ? 'Start New Game' : 'Restart'}
                </button>
                <div>{winner !== null ? `Player ${winner} wins` : null}</div>
                <div>{steps === size * size && winner === null ? 'Draw' : null}</div>
            </div>
    	);
    }
}

export default TicTacToe;
