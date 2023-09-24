package nnet.matrix.net;

import mxlib.excep.MxlibInvalidMatrixOp;
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
	private NNetMatrix bias = null;
	private ActivationFunction actvFunc = null;
	private Layer inputLayer = null;
	private Layer outputLayer = null;
	private int batchSize;
	private int numNodes;
	private int debugLevel = 0;
	private boolean useSoftMaxActivation = false;
	
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
	
	public int getDebugLevel() {
		return debugLevel;
	}

	public void setDebugLevel(int debugLevel) {
		this.debugLevel = debugLevel;
	}
	
	public void enableBias() {
		bias = NNetMatrix.createRandomMatrix(1,this.getNumNodes(),-1,+1) ;
	}

	public void enableBias(double min, double max) {
		bias = NNetMatrix.createRandomMatrix(1,this.getNumNodes(),min,max) ;
	}

	public boolean isUseSoftMaxActivation() {
		return useSoftMaxActivation;
	}

	public void setUseSoftMaxActivation(boolean useSoftMaxActivation) {
		this.useSoftMaxActivation = useSoftMaxActivation;
	}

	public void init() {
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			weights = NNetMatrix.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes()) ;
			useSoftMaxActivation = NNetParameters.getInstance().isUseSoftMax();
			break;
		case HIDDEN_LAYER:
			weights = NNetMatrix.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes()) ;
			break;
		}
	}
	
	/**
	 * Do a feed forward. 
	 * @throws MxlibInvalidMatrixOp 
	 */
	public void feedForward() throws MxlibInvalidMatrixOp {
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			hValues = inputLayer.aValues.dot(weights).bias(bias);
			aValues = (useSoftMaxActivation)? hValues.getSoftMax() : hValues;
			break;
		case HIDDEN_LAYER:
			hValues = inputLayer.hValues.dot(weights).bias(bias);
			aValues = hValues.getActivation(actvFunc);
			break;
		}	
		
		if (debugLevel>1)
		{
			hValues.print(layerType + ": hvalues");
			
			if (!isInputLayer()) {
				aValues.print(layerType + ": avalues");
				weights.print(layerType + ": weights");
			}
			
			if (bias!=null)
				bias.print(layerType + ": bias");
			
			if (isOutputLayer()) {
				System.out.println("Loss=" + getLoss());
			}
		}
	}

	

	
	public void backPropigation() throws MxlibInvalidMatrixOp {
		
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			aGradient = aValues.subtract(tValues).scale(2.0);	
			wGradient = inputLayer.aValues.transpose().dot(aGradient);	
			weights   = weights.subtract(wGradient.scale(NNetParameters.getInstance().getLearningRate()));
			if (bias!=null) 
				bias = bias.subtract(wGradient.getRowVector(0).scale(NNetParameters.getInstance().getLearningRate()));
			if (debugLevel>0)
				System.out.println("Loss=" + getLoss() + " Bias=" + ((bias!=null)? bias.toString("") : "N\\A"));
			break;
		case HIDDEN_LAYER:
			aGradient = outputLayer.aGradient.dot(outputLayer.weights.transpose());
			hGradient = aGradient.getGradient(actvFunc);
			wGradient = inputLayer.aValues.transpose().dot(hGradient);	
			weights   = weights.subtract(wGradient.scale(NNetParameters.getInstance().getLearningRate()));
			if (bias!=null) 
				bias = bias.subtract(wGradient.getRowVector(0).scale(NNetParameters.getInstance().getLearningRate()));
			if (debugLevel>0)
				System.out.println("Bias=" + ((bias!=null)? bias.toString("") : "N\\A"));
			break;
		}
	}
	
	/**
	 * The loss function for output layer else zero, Use mean squared error
	 * @return
	 * @throws MxlibInvalidMatrixOp 
	 */
	public double getLoss() throws MxlibInvalidMatrixOp {
		
		double ret = 0;
		if (isOutputLayer()) {
			double sumsqrs = hValues.subtract(tValues).squareElement().sum();
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
