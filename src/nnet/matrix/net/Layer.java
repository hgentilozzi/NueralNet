package nnet.matrix.net;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.matrix.NNetPerceptron;
import nnet.matrix.acvt.ActivationFunction;

public class Layer {
	private NNetPerceptron hValues = null;
	private NNetPerceptron aValues = null;
	private NNetPerceptron tValues = null;    // target values only for last layer
	private NNetPerceptron hGradient = null;
	private NNetPerceptron aGradient = null;
	private NNetPerceptron weights = null;
	private NNetPerceptron wGradient = null;
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
	
	public void setInputData(NNetPerceptron batch) {
		
		batchSize = batch.getRows();		
		hValues = aValues = batch;		
	}
	
	public void setExpectedData(NNetPerceptron exptectedResults) {		
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
			weights = NNetPerceptron.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes()) ;
			break;
		case HIDDEN_LAYER:
			weights = NNetPerceptron.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes()) ;
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

		if (actvFunc!=null)
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
	 * The loss function for output layer else zero
	 * @return
	 */
	public double getLoss() {
		return (tValues==null || aValues==null)? 0 : hValues.subtract(tValues).sum();
	}
	
	/**
	 * Set weights externally. Used for unit testing only
	 * @param weights
	 */
	public void setWeight(NNetPerceptron weights) {
		this.weights = weights;
	}
	
	
	public int getBatchSize() {
		return batchSize;
	}
	
	public int getNumNodes() {
		return numNodes;
	}
	
	public NNetPerceptron getOutputValues() {
		return aValues;
	}
	

}
