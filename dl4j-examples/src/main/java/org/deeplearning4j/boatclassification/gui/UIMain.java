package org.deeplearning4j.boatclassification.gui;

import java.io.File;
import java.util.concurrent.Executors;

import javax.swing.JFrame;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class UIMain {
	  private static Logger log = LoggerFactory.getLogger(UIMain.class);
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
	    File locationToSave = new File("boat-model.zip");
	    // Check for presence of saved model
	    if (locationToSave.exists()) {
	      log.info("Saved Model Found!");
	    } else {
	      log.error("File not found!");
	      log.error("This example depends on running MnistImagePipelineExampleSave, run that example first");
	      System.exit(0);
	    }
	    MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		JFrame mainFrame = new JFrame();
        UI ui = new UI();
        Executors.newCachedThreadPool().submit(() -> {
            try {
                ui.initUI(model);
            } catch (Exception e) {
                e.printStackTrace();
                throw new RuntimeException(e);
            } finally {
                mainFrame.dispose();
            }
        });
	}
}
