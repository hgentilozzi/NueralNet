package nnet.main.cfg;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import com.fasterxml.jackson.databind.ObjectMapper;

import nnet.matrix.data.NNetBatchDataFile;
import nnet.matrix.data.NnetBatchDataIntf;


public class NNetConfig {
	public String name;
	public String desc;
	public Double  learningrate;
	public Double  bias;
	public String trainingfilename;
	public String testfilename;
	public NNetCfgLayer[] layers;
	
	private NNetBatchDataFile trainingDB; 
	private NNetBatchDataFile testDB; 
	
	
	public NNetConfig() {
		trainingDB = null;
		testDB = null;
	}
	
	public static NNetConfig load(String fileName) {
		NNetConfig ret = null;
		
		String json;
		try {
			json = Files.readString(Paths.get(fileName));
			ObjectMapper mapper = new ObjectMapper();
			ret = mapper.readValue(json, NNetConfig.class);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return ret;
	}

	public NnetBatchDataIntf trainingData() {
		if (trainingDB==null) {
			trainingDB = new NNetBatchDataFile(trainingfilename, layers[0].nodes, layers[layers.length-1].nodes);
		}
		return trainingDB;
	}

	public NnetBatchDataIntf testData() {
		if (testDB==null) {
			testDB = new NNetBatchDataFile(testfilename, layers[0].nodes, layers[layers.length-1].nodes);
		}
		return testDB;
	}

}
