import * as tf from '@tensorflow/tfjs';
import { loadFrozenModel } from '@tensorflow/tfjs-converter';
import { h, render, Component } from 'preact';

/**
 * Use a TensorFlow.js model to predict the digit drawn into a canvas.
 *
 * Returns an array of confidence scores for each of the 10 digits.
 */
async function predictDigit(model, canvas) {
  // Extract the image from the canvas and preprocess it to match the
  // training data.
  //
  // TODO - Apply the image centering preprocessing step described at
  // http://yann.lecun.com/exdb/mnist/.
  const prediction = tf.tidy(() => {
    const channels = 1;
    const image = tf.div(tf.fromPixels(canvas, channels).asType('float32'), 255.0);
    const imageSize = 28;
    const scaledImage = tf.image.resizeBilinear(image, [imageSize, imageSize]);
    const flattenedImage = scaledImage.reshape([1, imageSize * imageSize]);

    // The model was trained to process batches of 32 images at a time.
    // Here we pad the single input image and then only use the first prediction.
    const batchSize = 32;
    const imageBatch = tf.pad(flattenedImage, [[0, batchSize - 1], [0, 0]]);

    const batchPredictions = model.execute({ image_input: imageBatch });
    return tf.slice(batchPredictions, [0, 0], [1, 10]).reshape([10]);
  });

  const predictionArray = Array.from(await prediction.data());
  prediction.dispose();
  return predictionArray;
}

/**
 * A bar displaying the score assigned to a class.
 */
function ScoreBar({ min, max, score, width }) {
  const percentage = Math.round((score / (max - min)) * 100);
  const style = {
    width,
    height: '100%',
    marginTop: 0,
    marginBottom: 5,
    background: `linear-gradient(to right, blue ${percentage}%, white ${percentage}%)`,
  };
  return <div style={style} title={score}></div>;
}

/**
 * A list of class labels and confidence scores.
 */
function ClassScoreList({ labels, scores }) {
  const min = Math.min(...scores);
  const max = Math.max(...scores);

  return <div class="ClassScoreList">
    {scores.map((score, index) =>
      <div class="ClassScoreList__item">
        <div class="ClassScoreList__label">{labels[index]}</div>
        ({score.toFixed(2)})
        <ScoreBar width={50} min={min} max={max} score={score}/>
      </div>
    )}
  </div>;
}

/**
 * Draw a series of points from mouse or touch events as a continous path.
 */
function drawPenPath(canvas, points) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineWidth = 30;
  ctx.strokeStyle = 'white';

  let mouseDown = false;
  ctx.beginPath();
  points.forEach((maybePoint) => {
    if (!maybePoint) {
      mouseDown = false;
      return;
    }
    const [x, y] = maybePoint;
    if (mouseDown) {
      ctx.lineTo(x, y);
    } else {
      ctx.moveTo(x, y);
    }
    mouseDown = true;
  });
  ctx.stroke();
}

/**
 * A component which lets the user draw handwritten digits and then
 * classifies them.
 */
class DigitApp extends Component {
  constructor(props) {
    super(props);

    this.canvasRef = null;
    this.state = {
      pathPoints: [],
      scores: null,
    };

    this._clear();
  }

  _clear() {
    const canvas = this.canvasRef;
    if (canvas) {
      drawPenPath(canvas, []);
    }
    this.setState({ pathPoints: [], scores: Array(10).fill(0) });
  }

  async _drawAndPredictDigit(event) {
    if (event.buttons === 0) {
      return;
    }
    const pathPoints = [...this.state.pathPoints];
    pathPoints.push([event.clientX, event.clientY]);
    drawPenPath(this.canvasRef, pathPoints);
    this.setState({
      pathPoints,
    });

    try {
      const scores = await predictDigit(this.props.model, event.target);
      this.setState({ scores });
    } catch (err) {
      console.error('Error predicting digit', err);
      this._clear();
    }
  };

  _recordMouseUp() {
    const pathPoints = [...this.state.pathPoints];
    pathPoints.push(null);
    this.setState({ pathPoints });
  }

  componentDidMount() {
    this._clear();
  }

  render() {
    const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    const scores = this.state.scores;
    const model = this.props.model;

    return <div>
      <div class="DigitApp__view">
        <canvas
          class="DigitApp__canvas"
          width={300} height={300}
          onMouseMove={event => this._drawAndPredictDigit(event)}
          onMouseUp={() => this._recordMouseUp()}
          ref={el => this.canvasRef = el} />
        <ClassScoreList labels={labels} scores={scores} />
      </div>
      <button class="DigitApp__clear-btn" onClick={() => this._clear()}>Clear</button>
    </div>
  }
}

/**
 * Load the TensorFlow.js model and initialize the UI.
 */
async function init() {
  tf.setBackend('cpu');

  const modelUrl = 'saved-model/tensorflowjs_model.pb';
  const weightsUrl = 'saved-model/weights_manifest.json';

  const model = await loadFrozenModel(modelUrl, weightsUrl);
  const appEl = document.querySelector('#app');

  render(<DigitApp model={model}/>, appEl);
}

init();
