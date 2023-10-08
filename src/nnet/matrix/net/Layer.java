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
	private double learningRate = 0.1;
	
	private LayerType layerType; 

	public Layer(LayerType layerType, int numNodes) {
		this(layerType,numNodes,null);
	}

	public Layer(LayerType layerType, int numNodes,ActivationFunction actvFunc) {
		this.layerType = layerType;
		this.numNodes = numNodes;
		this.actvFunc = (actvFunc==null)? ActivationFunction.LINEAR : actvFunc;
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

	public void setBias(double min, double max) {
		bias = NNetMatrix.createRandomMatrix(1,this.getNumNodes(),min,max) ;
	}

	public void setBias(double value) {
		bias = NNetMatrix.createRandomMatrix(1,this.getNumNodes(),value) ;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public void init() {
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			weights = NNetMatrix.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes(),-1,1) ;
			break;
		case HIDDEN_LAYER:
			weights = NNetMatrix.createRandomMatrix(inputLayer.getNumNodes(),this.getNumNodes(),-1,1) ;
			break;
		}
	}
	
	/**
	 * Do a feed forward. 
	 * @throws MxlibInvalidMatrixOp 
	 */
	public void feedForward() throws MxlibInvalidMatrixOp {
		if (layerType==LayerType.INPUT_LAYER)
		{
			if (debugLevel>0)
			{
				aValues.print(layerType + ": avalues");
			}
			
			aValues = hValues;
			return;
		}

		hValues = inputLayer.aValues.dot(weights).bias(bias);
		aValues = actvFunc.f(hValues);
		
		if (debugLevel>1)
		{
			weights.print(layerType + ": weights");
			hValues.print(layerType + ": hvalues");
			aValues.print(layerType + ": avalues");
			if (bias!=null)
				bias.print(layerType + ": bias");
			if (isOutputLayer())
				tValues.print(layerType + ": tvalues");
		}

	}

	

	
	public void backPropigation() throws MxlibInvalidMatrixOp {
		
		switch (layerType) {
		case INPUT_LAYER:
			break;
		case OUTPUT_LAYER:
			aGradient = aValues.subtract(tValues).scale(2.0);	
			hGradient = actvFunc.grad(aGradient);
			wGradient = inputLayer.aValues.transpose().dot(hGradient).scale(learningRate);	
			weights   = weights.subtract(wGradient);
			if (bias!=null) 
				bias = bias.subtract(wGradient.getRowVector(0)).scale(learningRate);
			break;
		case HIDDEN_LAYER:
			aGradient = outputLayer.aGradient.dot(outputLayer.weights.transpose());
			hGradient = actvFunc.grad(aGradient);

			wGradient = inputLayer.aValues.transpose().dot(hGradient).scale(learningRate);	
			weights   = weights.subtract(wGradient);
			if (bias!=null) 
				bias = bias.subtract(wGradient.getRowVector(0)).scale(learningRate);
			break;
		}
		
		
		if (debugLevel>1)
		{
			
			if (!isInputLayer()) {
				aGradient.print(layerType + ": aGradient");
				hGradient.print(layerType + ": hGradient");
				wGradient.print(layerType + ": wGradient");
				weights.print(layerType + ": weights");
				
				if (bias!=null)
					bias.print(layerType + ": bias");
			}			
		}
		
		if (debugLevel>0)
		{
			if (isOutputLayer()) {
				System.out.println("Loss=" + getLoss());
			}
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
			ret = (sumsqrs / (hValues.getRows()*hValues.getCols()));
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
