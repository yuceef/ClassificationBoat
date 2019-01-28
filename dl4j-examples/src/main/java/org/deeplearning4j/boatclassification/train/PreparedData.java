package org.deeplearning4j.boatclassification.train;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class PreparedData {
	private File csvFile;
	private String dataDir;
	public PreparedData(String csvFile, String dataDir) {
		super();
		this.csvFile = new File(csvFile);
		this.dataDir = dataDir;
		File directory = new File(dataDir+"Notfound");
	    if (! directory.exists()){
	        directory.mkdir();
	    }
	    directory = new File(dataDir+"Found");
	    if (! directory.exists()){
	        directory.mkdir();
	    }
	}
	public boolean organizedData() {
		try {
			BufferedReader b = new BufferedReader(new FileReader(csvFile));
			String readLine = "";
	        while ((readLine = b.readLine()) != null) {
	        	String[] info = readLine.split(",");
	        	if(info.length == 1) moveImage(info[0],"Notfound");
	        	else moveImage(info[0],"Found");
	        }
	        b.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return true;
	}
	private void moveImage(String image, String dir) {
		// TODO Auto-generated method stub
		try {
			Files.move 
			        (Paths.get(dataDir+image),  
			        		Paths.get(dataDir+dir+"\\"+image));
		} catch (IOException e) {
			// TODO Auto-generated catch block
		}
	}
	
}
