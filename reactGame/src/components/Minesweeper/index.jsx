import React from 'react';
import _ from 'lodash';
import propTypes from 'prop-types';

import * as styles from './index.css';

const FLAG = '⚐';
const INCORRECT_FLAG = '⦻';
const MINE = '○';

export class Minesweeper extends React.Component {
    static propTypes = {
    	minesNum: propTypes.number.isRequired,
        height: propTypes.number.isRequired,
        width: propTypes.number.isRequired,
    }

    constructor(...args) {
    	super(...args);
    	const { height, width } = this.props;
    	this.state = {
    		boxLeftClicked: 0,
    		boxRightClicked: 0,
    		display: Array(height).fill(null).map(row => new Array(width).fill(null)),
    		fail: false,
    		mines: this.sampleMines(),
    		time: 0,
    		timeId: null,
    		win: false,
    	}
    }

    componentDidUpdate() {
    	const { timeId, win, fail } = this.state;
    	if (timeId && (win || fail) ) {
    		clearInterval(timeId);
    		this.setState({
    			timeId: null,
    		});
    		if (win) { this.updateWin(); }
    	} 
    }

    sampleMines = () => {
    	const { minesNum, height, width } = this.props;
    	const sampleMines = _.sampleSize([...Array(width * height).keys()], minesNum);
        const mines = Array(height).fill(null).map(row => new Array(width).fill(false));
        
        sampleMines.forEach((mine) => {
        	mines[parseInt(mine / width, 0)][mine % width] = true;
        });
        
        return mines;
    }

    checkMines = (rowIdx, colIdx) => {
    	const { height, width } = this.props;
        const { mines } = this.state;
        let minesCount = 0;

        minesCount += rowIdx > 0 && mines[rowIdx - 1][colIdx] ? 1 : 0;
        minesCount += rowIdx > 0 && colIdx > 0 && mines[rowIdx - 1][colIdx - 1] ? 1 : 0;
        minesCount += rowIdx > 0 && colIdx < width - 1 && mines[rowIdx - 1][colIdx + 1] ? 1 : 0;
        minesCount += rowIdx < height - 1 && mines[rowIdx + 1][colIdx] ? 1 : 0;
        minesCount += rowIdx < height - 1 && colIdx > 0 && mines[rowIdx + 1][colIdx - 1] ? 1 : 0;
        minesCount += rowIdx < height - 1 && colIdx < width - 1 && mines[rowIdx + 1][colIdx + 1] ? 1 : 0;
        minesCount += colIdx > 0 && mines[rowIdx][colIdx - 1] ? 1 : 0;
        minesCount += colIdx < width - 1 && mines[rowIdx][colIdx + 1] ? 1 : 0;
        
        return minesCount;
    }

    handleClick = (rowIdx, colIdx) => {
    	const { mines, display, fail, win, timeId } = this.state;
    	const { height, width, minesNum } = this.props;
        if (!timeId) {
        	this.setState({
        		timeId: setInterval(() =>
        			this.setState((prevState) => {
        				return { time: prevState.time + 1 };
        			}), 1000
        		)
        	});
        }

    	if (!fail && !win && display[rowIdx][colIdx] === null){
    	    if (mines[rowIdx][colIdx]) {
    	    	this.setState({
    	    		fail: true,
    	    	})
    	    	this.updateFail();
    	    } else {
    	    	display[rowIdx][colIdx] = this.checkMines(rowIdx, colIdx);
    	    	this.setState((prevState) => {
    	    		return {
    	    			display,
                        boxRightClicked: prevState.boxRightClicked + 1,
                        win: (prevState.boxRightClicked + 1) === (height * width - minesNum),
    	    		};
    	    	});
    	    }
    	}
    }

    handleContextMenu = (event, rowIdx, colIdx) => {
    	event.preventDefault();
    	const { boxLeftClicked, display, fail, win } = this.state;
    	const { minesNum } = this.props;

        if (!fail && !win){
        	// reverse identified flag
    	    if (display[rowIdx][colIdx] === FLAG) {
                display[rowIdx][colIdx] = null;
                this.setState((prevState) => {
                	return {
                		display,
                		boxLeftClicked: prevState.boxLeftClicked - 1,
                	}
                });
            // do nothing if box is alredy rightclicked
    	    } else if (display[rowIdx][colIdx] === null && boxLeftClicked < minesNum) {
    	    	display[rowIdx][colIdx] = FLAG;
                this.setState((prevState) => {
                	return {
                	    display,
                	    boxLeftClicked: prevState.boxLeftClicked + 1,
                	};
                });
    	    }        	
        }
    }

    handleRestart = () => {
    	const { height, width } = this.props;
    	const { timeId } = this.state;
    	if (timeId) { clearInterval(timeId) }
    	this.setState({
    		boxRightClicked: 0,
    		display: Array(height).fill(null).map(row => new Array(width).fill(null)),
    		fail: false,
    		mines: this.sampleMines(),
    		boxLeftClicked: 0,
    		time: 0,
    		timeId: null,
    		win: false,
    	})
    }

    updateWin = () => {
    	const { display, mines } = this.state;
    	const updatedDisplay = display.map((row, rowIdx) => {
    		return row.map((box, colIdx) => mines[rowIdx][colIdx] ? FLAG : box)
    	});
    	this.setState({ display: updatedDisplay });
    }

    updateFail = () => {
    	const { display, mines } = this.state;
    	const updatedDisplay = display.map((row, rowIdx) => {
    		return row.map((box, colIdx) => {
    			return box === FLAG ?
    			    (mines[rowIdx][colIdx] ? box : INCORRECT_FLAG) :
    			    (mines[rowIdx][colIdx] ? MINE : box);
    		})
    	});
    	this.setState({ display: updatedDisplay });    	
    }

    render() {
    	const { display, fail, win, time, boxLeftClicked } = this.state;
    	const { minesNum } = this.props;
    	return(
            <div>
                <div>Time: {time} seconds</div>
                <div>{minesNum - boxLeftClicked} mines left</div>
                {display.map((row, rowIdx) =>
                    <div
                      key={`row-${rowIdx}`}
                      className={styles.row}
                    >
                        {row.map((box, colIdx) =>
                            <div
                              key={`box-${colIdx}`}
                              className={styles.box}
                              onClick={() => this.handleClick(rowIdx, colIdx)}
                              onContextMenu={(event) => this.handleContextMenu(event, rowIdx, colIdx)}
                            >
                                <span 
                                  style={{ margin: 'auto' }}
                                  className={(box === FLAG || box === MINE || box === INCORRECT_FLAG) ? styles.red : null}
                                >
                                    {box}
                                </span>
                            </div>
                        )}
                    </div>
                )}
                <button type="button" onClick={this.handleRestart}>
                    {win || fail ? 'Start New Game' : 'Restart'}
                </button>
                <div>{win ? 'You Win!!' : null}</div>
                <div>{fail ? 'Ooops!!' : null}</div>
            </div>
    	)
    }
}

export default Minesweeper;
