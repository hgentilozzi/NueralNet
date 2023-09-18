package nnet.matrix.net;

import java.util.ArrayList;
import java.util.List;

import nnet.exception.NNetInvalidMatrixOp;
import nnet.exception.NNetInvalidNetwork;
import nnet.matrix.NNetPerceptron;
import nnet.matrix.acvt.ActivationFunction;

public class Network {
	private int numLayers;
	private List<Layer> layers;
	private Layer inputLayer;
	private Layer outputLayer;
	private TrainingStats stats;
	

	public Network(ActivationFunction acvtFunc,int ... layersizes) throws NNetInvalidNetwork {
		this.numLayers = layersizes.length;
		
		// Must have a least 2 layers, input, hidden, and output
		if (numLayers<2)
			throw new NNetInvalidNetwork("Network validate: must have a least 3 layers");	
		

		// Create the layers
		layers = new ArrayList<Layer>();
		layers.add(new Layer(LayerType.INPUT_LAYER,1,acvtFunc));
		layers.add(new Layer(LayerType.HIDDEN_LAYER,2,acvtFunc));
		layers.add(new Layer(LayerType.OUTPUT_LAYER,1));

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
		
	}

	public void train(NNetPerceptron iData, NNetPerceptron oData, int iterations) throws NNetInvalidMatrixOp 
	{
		train(iData,oData,iterations,0);
	}
	
	public void train(NNetPerceptron iData, NNetPerceptron oData, int iterations,double tolerance) throws NNetInvalidMatrixOp 
	{
		stats.clear();
		
		inputLayer.setInputData(iData);
		outputLayer.setExpectedData(oData);

		for (int iter=0;iter<iterations;iter++)
		{
			for (int i=0;i<layers.size();i++)
				layers.get(i).feedForward();

			for (int i=layers.size()-1;i>=0;i--)
				layers.get(i).backPropigation();	

			stats.totalIterations++; 

			double loss = getLoss();
			if (tolerance>0 && loss>-tolerance && loss < tolerance)
					break;
		}

		stats.finalLoss = getLoss();
		stats.finalResult = outputLayer.getOutputValues();

	}
	
	
	public NNetPerceptron predict(NNetPerceptron iData) throws NNetInvalidMatrixOp
	{
		inputLayer.setInputData(iData);
		
		for (int i=0;i<layers.size();i++)
			layers.get(i).feedForward();
		
		return outputLayer.getOutputValues();
	}	
	
	public double getLoss() {
		return outputLayer.getLoss();
	}

	public TrainingStats getStats() {
		return stats;
	}
	
	
	
	
}
