import React from 'react';
import propTypes from 'prop-types';

import * as styles from './index.css';

class Connect4 extends React.Component {
    static propTypes = {
        connect: propTypes.number.isRequired,
        height: propTypes.number.isRequired,
        width: propTypes.number.isRequired,
    }
    
    constructor(...args) {
        super(...args);
        const { height, width } = this.props;
        this.state = {
            boxes: Array(height).fill(0).map(r => Array(width).fill(null)),
            steps: 0,
            turn: 1,
            winner: null,
        }
    }

    checkWinner = (rowIdx, boxIdx, turn) => {
        const { connect, height, width } = this.props;
        const { boxes } = this.state;
        let down = 0;
        let right = 0;
        let left = 0;
        let leftTop = 0;
        let rightBottom = 0;
        let rightTop = 0;
        let leftBottom = 0;

        // check vertical
        let counter = 1;
        while (counter < connect && (rowIdx + counter) < height) {
            if (boxes[rowIdx + counter][boxIdx] === turn) {
                down += 1;
                counter += 1;
            } else {
                break;
            }
        }
        if (down === (connect - 1)) {
            return turn;
        }

        // check right
        counter = 1;
        while (counter < connect && (boxIdx + counter) < width) {
            if (boxes[rowIdx][boxIdx + counter] === turn) {
                right += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (right === (connect - 1)) {
            return turn;
        }
        
        // check left
        counter = 1;
        while (counter < connect && (boxIdx - counter) >= 0) {
            if (boxes[rowIdx][boxIdx - counter] === turn) {
                left += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (right + left === (connect - 1)) {
            return turn;
        }        

        // check left diagonal
        counter = 1;
        while (counter < connect && (boxIdx - counter) >= 0 && (rowIdx - counter) >= 0) {
            if (boxes[rowIdx - counter][boxIdx - counter] === turn) {
                leftTop += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        counter = 1;
        
        while (counter < connect && (boxIdx + counter) < width && (rowIdx + counter) < height) {
            if (boxes[rowIdx + counter][boxIdx + counter] === turn) {
                rightBottom += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (leftTop + rightBottom >= (connect - 1)) {
            return turn
        }

        // check right diagonal
        counter = 1;
        while (counter < connect && (boxIdx + counter) < width && (rowIdx - counter) >= 0) {
            if (boxes[rowIdx - counter][boxIdx + counter] === turn) {
                rightTop += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        counter = 1;
        
        while (counter < connect && (boxIdx - counter) >= 0 && (rowIdx + counter) < height) {
            if (boxes[rowIdx + counter][boxIdx - counter] === turn) {
                leftBottom += 1;
                counter += 1;
            } else {
                break;
            }            
        }
        if (rightTop + leftBottom >= (connect - 1)) {
            return turn
        }

        return null;
    }

    handleClick = (rowIdx, boxIdx) => {
        const { boxes, turn, winner } = this.state;
        const { height } = this.props;
        // check if it is a valid box and is still empty
        if ((rowIdx === (height - 1) || boxes[rowIdx + 1][boxIdx]) && !boxes[rowIdx][boxIdx] && !winner) {
            boxes[rowIdx][boxIdx] = turn;
            this.setState((prevState) => {
              return {
                boxes,
                steps: prevState.steps + 1,
                turn: turn === 1 ? 2 : 1,
                winner: this.checkWinner(rowIdx, boxIdx, turn),
            }})
        }
    }

    handleRestart = () => {
        const { height, width } = this.props;
        this.setState({
            boxes: Array(height).fill(0).map(r => Array(width).fill(null)),
            steps: 0,
            turn: 1,
            winner: null,
        })
    }
  
    render() {
        const { boxes, winner, steps } = this.state;
        const { width, height } = this.props;
        return (
            <div>
                {boxes.map((row, rowIdx) => 
                    <div key={`row-${rowIdx}`} className={styles.row}>
                        {row.map((box, boxIdx) => 
                            <div 
                              key={`box-${boxIdx}`}
                              className={styles.box}
                              onClick={() => this.handleClick(rowIdx, boxIdx)}
                            >
                                <div className={`${styles.font} ${box === 1 ? styles.font0 : styles.font1}`}>
                                    {box ? '◯' : null}
                                </div>
                            </div>
                        )}
                    </div>
                )}
                <button type="button" onClick={this.handleRestart}>
                    {steps === width * height || winner ? 'Start New Game' : 'Restart'}
                </button>
                <div>{winner ? `Player ${winner} wins` : null}</div>
                <div>{steps === width * height && !winner ? 'Draw' : null}</div>
            </div>
        );
    }
}

export default Connect4;
