package nnet.matrix.net;

import java.util.ArrayList;
import java.util.List;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.exception.NNetInvalidNetwork;
import nnet.matrix.NNetMatrix;
import nnet.matrix.acvt.ActivationFunction;
import nnet.matrix.data.*;

public class Network {
	private int numLayers;
	private List<Layer> layers;
	private Layer inputLayer;
	private Layer outputLayer;
	private TrainingStats stats;
	private int debugLevel;
	private ActivationFunction acvtFunc;
	

	public Network(int ... layersizes) throws NNetInvalidNetwork {
		this.numLayers = layersizes.length;
		this.acvtFunc = NNetParameters.getInstance().getActivationFunction();
		
		// Must have a least 2 layers, input, hidden, and output
		if (numLayers<2)
			throw new NNetInvalidNetwork("Network validate: must have a least 3 layers");	
		

		// Create the layers
		layers = new ArrayList<Layer>();
		
		layers.add(new Layer(LayerType.INPUT_LAYER,layersizes[0],acvtFunc));
		
		for (int nl = 1; nl < numLayers-1; nl++)
			layers.add(new Layer(LayerType.HIDDEN_LAYER,layersizes[nl],acvtFunc));
		
		layers.add(new Layer(LayerType.OUTPUT_LAYER,layersizes[numLayers-1]));

		// Wire them together
		for (int i=0;i<layers.size();i++)
		{
			if (i>0)
				layers.get(i).setInputLayer(layers.get(i-1));
			if (i<layers.size()-1)
				layers.get(i).setOutputLayer(layers.get(i+1));
		}
		
		// Set initial weights
		layers.stream().forEach(l -> l.init());
		
		// For convenience
		inputLayer = layers.get(0);
		outputLayer = layers.get(layers.size()-1);
		
		stats = new TrainingStats();
		debugLevel = 0;
		
	}

	public int getDebugLevel() {
		return debugLevel;
	}

	public void setDebugLevel(int debugLevel) {
		this.debugLevel = debugLevel;
		layers.stream().forEach(l -> l.setDebugLevel(debugLevel));
	}
	
	public void enableBias() {
		for (int nl = 1; nl < numLayers; nl++) 
			layers.get(nl).enableBias();
	}

	/**
	 * Train using an input stream of and train each batch
	 * @param batchData  - all input data and expected results
	 * @param iterations
	 * @throws Exception 
	 */
	public void train(NnetBatchDataIntf batchData, int iterations) throws Exception 
	{
		stats.clear();
		for (int iter=0;iter<iterations;iter++)
		{
			batchData.reset();
			while (!batchData.atEof()) {
				NNetBatch b = batchData.nextBatch(1);
				train(b.inData(),b.expectedResults());
			}
		}
		stats.finalLoss = getLoss();
		stats.finalResult = outputLayer.getOutputValues();
	}
	
	public void train(NNetMatrix iData, NNetMatrix oData) throws MxlibInvalidMatrixOp 
	{
		
		inputLayer.setInputData(iData);
		outputLayer.setExpectedData(oData);

		for (int i=0;i<layers.size();i++)
			layers.get(i).feedForward();

		for (int i=layers.size()-1;i>=0;i--)
			layers.get(i).backPropigation();	
				
		if (debugLevel>1)
		{
			outputLayer.getOutputValues().print("OL->avalues");
			outputLayer.getWeights().print("OL->weights");
		}

		stats.totalIterations++; 
		stats.addLossResult(getLoss());

	}
	
	
	public NNetMatrix predict(NNetMatrix iData) throws MxlibInvalidMatrixOp
	{
		inputLayer.setInputData(iData);
		
		for (int i=0;i<layers.size();i++)
			layers.get(i).feedForward();
		
		return outputLayer.getOutputValues();
	}	
	
	public double getLoss() throws MxlibInvalidMatrixOp {
		return outputLayer.getLoss();
	}

	public TrainingStats getStats() {
		return stats;
	}
	
	
	
	
}
