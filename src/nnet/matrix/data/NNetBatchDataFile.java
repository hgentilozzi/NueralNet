package nnet.matrix.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import nnet.matrix.*;

/**
 * A batch based on a set of arrays
 */
public class NNetBatchDataFile implements NnetBatchDataIntf {
	private int batchSize;
	private int numInNodes;
	private int numOutNodes;
	private String error;
	private BufferedReader reader;
	private String fileName;
	private String nextLine;

	public NNetBatchDataFile(String fileName) throws IOException {
		this.error = "";
		this.fileName = fileName;
		this.reader = null;
		this.nextLine = null;
		
		// Open the file
		reader = new BufferedReader(new FileReader(fileName));
		
		// The first line is: "HEADER:", batch size,num input nodes, num output node
		String hdrline = reader.readLine();
		if (hdrline==null || !hdrline.startsWith("HEADER:")) {
			closeReader();
			reader = null;
			error = "First line of input file does not begin with \"HEADER:\"";
		}
			
	}

	private void closeReader() {
		if (reader!=null)
			try {
				reader.close();
			} catch (IOException e) {
			}
	}
	
	@Override
	public void reset() throws Exception {
		// Close the current file if already open
		closeReader();
		
		// Open the file
		reader = new BufferedReader(new FileReader(fileName));
		
		// The first line is: "HEADER:", batch size,num input nodes, num output node
		String hdrline = reader.readLine();
		if (hdrline==null || !hdrline.startsWith("HEADER:")) {
			closeReader();
			reader = null;
			error = "First line of input file does not begin with \"HEADER:\"";
		}
		else
		{
			// Parse the header line
			String[] parts = hdrline.split(",");
			if (parts.length<4)
			{
				closeReader();
				reader = null;
				error = "The header line is ill-formed";
			}
			else
			{
				batchSize = Integer.parseInt(parts[1]);
				numInNodes = Integer.parseInt(parts[2]);
				numOutNodes = Integer.parseInt(parts[3]);
			}
			
			// pre-read the first line of data
			nextLine = reader.readLine();
		}
	}


	@Override
	public boolean atEof() {
		return nextLine==null;
	}

	@Override
	public NNetBatch nextBatch() throws Exception {
		NNetBatch ret = null;
		List<String> batchlines = new ArrayList<String>();
		
		if (atEof())
			return null;
		
		batchlines.add(nextLine);
		int i=1;
		String l;
		// Read up to batchsize of rows
		while ((l=reader.readLine())!=null){
			batchlines.add(l);
			if (++i>batchSize)
				break;
		}
		
		// buffer the next line for eof checking
		nextLine = reader.readLine();
		
		// create the NNetMatrix for input and output
		double[][] indate = new double[batchlines.size()][numInNodes];
		double[][] outdate = new double[batchlines.size()][numOutNodes];
		int row = 0;

		for (String bstr : batchlines) {
			String[] parts = bstr.split(",");
			if (parts.length!=(numInNodes+numOutNodes)) {
				closeReader();
				reader = null;
				error = "The data line is ill-formed: line=" +l;
				throw new Exception(error);
			}
			
			for (int col=0;col<numInNodes;col++)
				indate[row][col] = Double.parseDouble(parts[col]);

			for (int col=0;col<numOutNodes;col++)
				outdate[row][col] = Double.parseDouble(parts[numInNodes+col]);

			ret = new NNetBatch(batchlines.size(),new NNetMatrix(indate), new NNetMatrix(outdate));
			
		}

		return ret;
	}

	@Override
	public boolean validate() {
		// Reader failed to open Error already set
		if (reader==null)
			return false;

		return true;
	}

	@Override
	public String getError() {
		return error;
	}


}