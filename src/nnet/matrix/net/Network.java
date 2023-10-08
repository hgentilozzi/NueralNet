package nnet.matrix.net;

import java.util.ArrayList;
import java.util.List;

import mxlib.excep.MxlibInvalidMatrixOp;
import nnet.exception.NNetInvalidNetwork;
import nnet.main.cfg.NNetConfig;
import nnet.main.cfg.NNetCfgLayer;
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
	private NNetConfig cfg;
	
	public Network(NNetConfig cfg) throws NNetInvalidNetwork {
		this.cfg = cfg;
		numLayers = this.cfg.layers.length;
		
		// Must have a least 2 layers, input and output
		if (numLayers<2)
			throw new NNetInvalidNetwork("Network validate: must have a least 2 layers");	
		
		// Create the layers
		layers = new ArrayList<Layer>();
		int lastLayerIdx = cfg.layers.length-1;
		for (int i=0;i<=lastLayerIdx;i++) {
			NNetCfgLayer cl = cfg.layers[i];
			if (i==0)
				layers.add(new Layer(LayerType.INPUT_LAYER,cl.nodes));
			else if (i==lastLayerIdx)
				layers.add(new Layer(LayerType.OUTPUT_LAYER,cl.nodes,ActivationFunction.getByType(cl.activationfunction)));
			else
				layers.add(new Layer(LayerType.HIDDEN_LAYER,cl.nodes,ActivationFunction.getByType(cl.activationfunction)));
			
			if (cfg.bias!=null)
				layers.get(i).setBias(cfg.bias);
			
		}
		

		// Wire them together
		for (int i=0;i<layers.size();i++)
		{
			if (i>0)
				layers.get(i).setInputLayer(layers.get(i-1));
			if (i<layers.size()-1)
				layers.get(i).setOutputLayer(layers.get(i+1));
			
			layers.get(i).setLearningRate(cfg.learningrate);
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
//	
//	public void enableBias() {
//		for (int nl = 1; nl < numLayers; nl++) 
//			layers.get(nl).enableBias();
//	}
//	
//	public void setBias(double value) {
//		for (int nl = 1; nl < numLayers; nl++) 
//			layers.get(nl).setBias(value);
//	}

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
			if (debugLevel>1)
				System.out.println("Training iterartion " + iter);
			batchData.reset();
			while (!batchData.atEof()) {
				NNetBatch b = batchData.nextBatch(1);
				train(b.inData(),b.expectedResults());
			}
			if (debugLevel>1)
				System.out.println("\n\n");

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
				
		if (debugLevel>0)
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
