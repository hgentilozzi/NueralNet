package nnet.matrix.net;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.ActivationFunction;

public class Layer {
	private NNetMatrix hValues = null;
	private NNetMatrix aValues = null;
	private NNetMatrix tValues = null;    // target values only for last layer
	private NNetMatrix hGradient = null;
	private NNetMatrix aGradient = null;
	private NNetMatrix weights = null;
	private NNetMatrix wGradient = null;
	private ActivationFunction actvFunc = null;
	private Layer inputLayer = null;
	private Layer outputLayer = null;
	private int batchSize;
	private int numNodes;
	
	private LayerType layerType; 

	public Layer(LayerType layerType, int numNodes) {
		this(layerType,numNodes,null);
	}

	public Layer(LayerType layerType, int numNodes,ActivationFunction actvFunc) {
		this.layerType = layerType;
		this.numNodes = numNodes;
		this.actvFunc = actvFunc;
	}
	
	public void setInputData(NNetMatrix batch) {
		
		batchSize = batch.getRows();		
		hValues = aValues = batch;		
	}
	
	public void setExpectedData(NNetMatrix exptectedResults) {		
		tValues = exptectedResults;
	}
	
	public Layer getInputLayer() {
		return inputLayer;
	}

	public void setInputLayer(Layer inputLayer) {
		this.inputLayer = inputLayer;
	}

	public Layer getOutputLayer() {
		return outputLayer;
	}

	public void setOutputLayer(Layer outputLayer) {
		this.outputLayer = outputLayer;
	}

	public void init() {
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			weights = NNetMatrix.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes()) ;
			break;
		case HIDDEN_LAYER:
			weights = NNetMatrix.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes()) ;
			break;
		}
	}
	
	/**
	 * Do a feed forward. 
	 * @throws NNetInvalidMatrixOp 
	 */
	public void feedForward() throws NNetInvalidMatrixOp {
		if (layerType==LayerType.OUTPUT_LAYER)
			return;

		//outputLayer.weights.print("ol_hv");
		
		if (layerType==LayerType.INPUT_LAYER)
			outputLayer.hValues = hValues.dot(outputLayer.weights);
		else
			outputLayer.hValues = aValues.dot(outputLayer.weights);

		//outputLayer.weights.print("ol_hv");

		if (actvFunc!=null && outputLayer.isHiddenLayer())
			outputLayer.aValues = outputLayer.hValues.getActivation(actvFunc);
		else
			outputLayer.aValues = outputLayer.hValues;	
	}
	
	public void backPropigation() throws NNetInvalidMatrixOp {
		
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			aGradient = aValues.subtract(tValues).scale(2.0);	
			wGradient = inputLayer.aValues.transpose().dot(aGradient);	
			weights   = weights.subtract(wGradient.scale(NNetParameters.getInstance().getLearningRate()));
			break;
		case HIDDEN_LAYER:
			aGradient = outputLayer.aGradient.dot(outputLayer.weights.transpose());
			hGradient = aGradient.getGradient(actvFunc);
			wGradient = inputLayer.aValues.transpose().dot(hGradient);	
			weights   = weights.subtract(wGradient.scale(NNetParameters.getInstance().getLearningRate()));
			break;
		}		
	}
	
	/**
	 * The loss function for output layer else zero, Use mean squared error
	 * @return
	 */
	public double getLoss() {
		
		double ret = 0;
		if (isOutputLayer()) {
			double sumsqrs = hValues.subtract(tValues).square().sum();
			ret = 1.0 / ((hValues.getRows()*hValues.getCols()) * sumsqrs);
		}
			
		return ret;
	}
	
	/**
	 * Set weights externally. Used for unit testing only
	 * @param weights
	 */
	public void setWeight(NNetMatrix weights) {
		this.weights = weights;
	}
	
	public NNetMatrix getWeights() {
		return weights;
	}

	public int getBatchSize() {
		return batchSize;
	}
	
	public int getNumNodes() {
		return numNodes;
	}
	
	public NNetMatrix getOutputValues() {
		return aValues;
	}
	
	public  boolean isInputLayer() {
		return  layerType ==  LayerType.INPUT_LAYER;
	}
	
	public  boolean isOutputLayer() {
		return  layerType ==  LayerType.OUTPUT_LAYER;
	}
	
	public  boolean isHiddenLayer() {
		return  layerType ==  LayerType.HIDDEN_LAYER;
	}
	

}
