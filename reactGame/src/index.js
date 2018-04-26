import React from 'react';
import ReactDOM from 'react-dom';
import Connect4 from './components/Connect4';
import registerServiceWorker from './registerServiceWorker';

ReactDOM.render(<Connect4 height={8} width={10} connect={4}/>, document.getElementById('root'));
registerServiceWorker();
