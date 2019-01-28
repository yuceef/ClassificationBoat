package org.deeplearning4j.boatclassification.gui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.UIManager;
import javax.swing.WindowConstants;
import javax.swing.plaf.FontUIResource;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class UI {
	private static Logger log = LoggerFactory.getLogger(UI.class);

	private JFrame mainFrame;
	private JPanel mainPanel;
	private static final int FRAME_WIDTH = 800;
	private static final int FRAME_HEIGHT = 800;
	private ImagePanel sourceImagePanel;
	private JLabel predictionResponse,statistic;
	private File selectedFile;
	private MultiLayerNetwork model;

	public UI() throws Exception {
		UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		UIManager.put("Button.font", new FontUIResource(new Font("Dialog", Font.BOLD, 12)));

	}

	public void initUI(MultiLayerNetwork model) throws Exception {
		this.model = model;
		// create main frame
		mainFrame = createMainFrame();

		mainPanel = new JPanel();
		mainPanel.setLayout(new GridBagLayout());

		JButton chooseButton = new JButton("Choisir une image");
        chooseButton.addActionListener(e -> {
            chooseFileAction();
            predictionResponse.setText("");
            statistic.setText("");
        });
		JButton predictButton = new JButton("Traiter l'image");

		fillMainPanel(chooseButton, predictButton);
        predictButton.addActionListener(e -> {
        	predict();
        });

		addSignature();

		mainFrame.add(mainPanel, BorderLayout.CENTER);
		mainFrame.setVisible(true);

	}

	private void predict() {
	    int height = 28;
	    int width = 28;
	    int channels = 3;
	    List<String> labelList = Arrays.asList("Found","Notfound");
    	// Use NativeImageLoader to convert to numerical matrix
	    NativeImageLoader loader = new NativeImageLoader(height, width, channels);
	    // Get the image into an INDarray
	    INDArray image;
		try {
			image = loader.asMatrix(selectedFile);
    	    // 0-255
    	    // 0-1
    	    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    	    scaler.transform(image);
    	    
    	    // Pass through to neural Net
    	    INDArray output = model.output(image);

    	    log.info("The file chosen was " + selectedFile);
    	    log.info("The neural nets prediction (list of probabilities per label)");
    	    //log.info("## List of Labels in Order## ");
    	    // In new versions labels are always in order
    	    log.info(output.toString());
    	    log.info(labelList.toString());
    	    String stat = "Found : "+(int)(output.getFloat(0)*100) +"% et NotFound : " + (int)(output.getFloat(1)*100)+"%";
    	    if (output.getFloat(0) >= output.getFloat(1)*1.5 ) {
                predictionResponse.setText("Found");
                predictionResponse.setForeground(Color.GREEN);
            } else if (output.getFloat(1) >= output.getFloat(0)*1.5 ) {
                predictionResponse.setText("NotFound");
                predictionResponse.setForeground(Color.GREEN);
            } else {
                predictionResponse.setText("Not Sure...");
                predictionResponse.setForeground(Color.RED);
            }
            statistic.setText(stat);

		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
    }

	private void fillMainPanel(JButton chooseButton, JButton predictButton) throws IOException {
		GridBagConstraints c = new GridBagConstraints();

		c.gridx = 1;
		c.gridy = 1;
		c.weighty = 0;
		c.weightx = 0;
		JPanel buttonsPanel = new JPanel(new FlowLayout());
		buttonsPanel.add(chooseButton);
		buttonsPanel.add(predictButton);
		mainPanel.add(buttonsPanel, c);

		c.gridx = 1;
		c.gridy = 2;
		c.weighty = 1;
		c.weightx = 1;
		sourceImagePanel = new ImagePanel();
		mainPanel.add(sourceImagePanel, c);

		c.gridx = 1;
		c.gridy = 3;
		c.weighty = 0;
		c.weightx = 0;
		predictionResponse = new JLabel();
		predictionResponse.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 32));
		mainPanel.add(predictionResponse, c);

		statistic = new JLabel();
		c.gridx = 1;
		c.gridy = 4;
		c.weighty = 0;
		c.weightx = 0;
		statistic.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 22));
		mainPanel.add(statistic, c);
	}

	public void chooseFileAction() {
		JFileChooser chooser = new JFileChooser();
		int action = chooser.showOpenDialog(null);
		if (action == JFileChooser.APPROVE_OPTION) {
			try {
				selectedFile = chooser.getSelectedFile();
				showSelectedImageOnPanel(new FileInputStream(selectedFile), sourceImagePanel);
			} catch (IOException e1) {
				throw new RuntimeException(e1);
			}
		}
	}

	private void showSelectedImageOnPanel(InputStream selectedFile, ImagePanel imagePanel) throws IOException {
		BufferedImage bufferedImage = ImageIO.read(selectedFile);
		imagePanel.setImage(bufferedImage);
	}

	private JFrame createMainFrame() {
		JFrame mainFrame = new JFrame();
		mainFrame.setTitle("Image Recognizer");
		mainFrame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
		mainFrame.setSize(FRAME_WIDTH, FRAME_HEIGHT);
		mainFrame.setLocationRelativeTo(null);
		mainFrame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosed(WindowEvent e) {
				System.exit(0);
			}
		});
		ImageIcon imageIcon = new ImageIcon("icon.png");
		mainFrame.setIconImage(imageIcon.getImage());

		return mainFrame;
	}

	private void addSignature() {
		JLabel signature = new JLabel("YHteam", JLabel.HORIZONTAL);
		signature.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 20));
		signature.setForeground(Color.BLUE);
		mainFrame.add(signature, BorderLayout.SOUTH);
	}
}
