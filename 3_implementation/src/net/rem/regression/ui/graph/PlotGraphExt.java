/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.ui.graph;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.awt.print.PageFormat;
import java.awt.print.Printable;
import java.awt.print.PrinterException;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import flanagan.plot.Plot;
import net.hudup.core.Util;
import net.hudup.core.logistic.UriAssoc;
import net.hudup.core.logistic.xURI;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.core.parser.TextParserUtil;

/**
 * This class is an extension of plot graph.
 * 
 * @author Loc Nguyen, Michael T Flanagan
 * @version 1.0
 *
 */
public class PlotGraphExt extends /*PlotGraph*/ PlotGraphFlanagan implements Graph {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Constructor with X data and Y data.
	 * @param xData specified X data.
	 * @param yData specified Y data.
	 */
	public PlotGraphExt(double[] xData, double[] yData) {
		super(xData, yData);
	}

	
	/**
	 * Constructor with data array.
	 * @param data specified data array.
	 */
	public PlotGraphExt(double[][] data) {
		super(data);
	}

	
	@Override
	public int print(Graphics g, PageFormat pf, int pageIndex)
			throws PrinterException {
		
		//assume the page exists until proven otherwise
		int retval = Printable.PAGE_EXISTS;
		
		//We only want to deal with the first page.
		//The first page is numbered '0'
		if (pageIndex > 0){
			retval = Printable.NO_SUCH_PAGE;
		}
		else {
			//setting up the Graphics object for printing
			g.translate((int)(pf.getImageableX()), (int)(pf.getImageableY()));
	    		//populate the Graphics object from HelloPrint's paint() method
			paint(g);
		}
	    return retval;
	}

	
	/**
	 * Painting method
	 * @param g graphics context.
	 * @param width specified with.
	 * @param height specified height.
	 */
	public void paint(Graphics g, int width, int height){

    	// Rescale - needed for redrawing if graph window is resized by dragging
    	double newGraphWidth = width;
    	double newGraphHeight = height;
    	double xScale = newGraphWidth/(double)this.graphWidth;
    	double yScale = newGraphHeight/(double)this.graphHeight;
    	rescaleX(xScale);
    	rescaleY(yScale);

    	// Call graphing method
    	graph(g);
	}

	
	@Override
	public void exportImage() {
		UriAssoc uriAssoc = Util.getFactory().createUriAssoc(xURI.create(new File(".")));
		xURI chooseUri = uriAssoc.chooseUri(this, false, new String[] {"png"}, new String[] {"PNG file"}, null, "png");
		if (chooseUri == null) {
			JOptionPane.showMessageDialog(
					this, 
					"Invalid URI", 
					"Image not exported", 
					JOptionPane.INFORMATION_MESSAGE);
			return;
		}
		
		BufferedImage image = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_ARGB);
		Graphics2D g = image.createGraphics();
		g.setColor(new Color(0, 0, 0));
		paint(g);
		try {
			ImageIO.write(image, "png", new File(chooseUri.getURI()));
			JOptionPane.showMessageDialog(
					this, 
					"Image exported successfully", 
					"Image exported successfully", 
					JOptionPane.INFORMATION_MESSAGE);
		}
		catch (IOException e1) {
			e1.printStackTrace();
		}
		
	}

	
	@Override
	public Rectangle getOuterBox() {
		return new Rectangle(0, 0, graphWidth, graphHeight);
	}


	@Override
	public void setupViewOption() {
		GraphViewOption opt = getViewOption();
		PlotGraphViewOptionDlg dlg = new PlotGraphViewOptionDlg(this, opt);
		if (dlg.getViewOption() != null && dlg.getViewOption().legends.length > 1) {
			setViewOption(dlg.getViewOption());
		}
		
	}

	
	@Override
	public GraphViewOption getViewOption() {
		GraphViewOption opt = new GraphViewOption();
		opt.legends = new String[] { xAxisLegend, yAxisLegend };
		
		int index = graphTitle.indexOf(":");
		if (index == -1)
			opt.graphTitle = graphTitle;
		else
			opt.graphTitle = graphTitle.substring(0, index);
		
		return opt;
	}
	
	
	/**
	 * Setting view option.
	 * @param opt view option.
	 */
	public void setViewOption(GraphViewOption opt) {
		if (opt == null)
			return;
		
		if (opt.legends.length > 1) {
			setXaxisLegend(opt.legends[0]);
			setYaxisLegend(opt.legends[1]);
		}
		if (opt.graphTitle != null && !opt.graphTitle.isEmpty())
			setGraphTitle(opt.graphTitle + ":" + getGraphFeature());
		
		this.repaint();
	}
	
	
	/**
	 * Getting graph feature.
	 * @return graph feature.
	 */
	public String getGraphFeature() {
		return "";
	}
	
	
}


/**
 * This class shows a view option dialog.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class PlotGraphViewOptionDlg extends JDialog {

	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Graph title.
	 */
	JTextField graphTitle = null;

	
	/**
	 * Graph legends.
	 */
	JTextField legends = null;
	
	
	/**
	 * Returned view option.
	 */
	GraphViewOption returnViewOption = null;
	
	
	/**
	 * Constructor with view option.
	 * @param comp parent component.
	 * @param option view option.
	 */
	public PlotGraphViewOptionDlg(Component comp, GraphViewOption option) {
		super (UIUtil.getDialogForComponent(comp), "Plot graph option", true);
		
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(400, 200);
		setLocationRelativeTo(UIUtil.getDialogForComponent(comp));
		
		setLayout(new BorderLayout());
		
		JPanel header = new JPanel(new BorderLayout());
		add(header, BorderLayout.NORTH);
		
		JPanel left = new JPanel(new GridLayout(0, 1));
		header.add(left, BorderLayout.WEST);
		
		left.add(new JLabel("Graph title: "));
		left.add(new JLabel("Legends: "));
		
		JPanel center = new JPanel(new GridLayout(0, 1));
		header.add(center, BorderLayout.CENTER);

		graphTitle = new JTextField();
		if (option != null && option.graphTitle != null)
			graphTitle.setText(option.graphTitle);
		else
			graphTitle.setText("");
		center.add(graphTitle);
		
		legends = new JTextField();
		if (option != null && option.legends != null && option.legends.length > 1) {
			StringBuffer buffer = new StringBuffer();
			for (int i = 0; i < option.legends.length; i++) {
				if (i > 0)
					buffer.append(",");
				
				buffer.append(option.legends[i]);
			}
			legends.setText(buffer.toString());
		}
		else
			legends.setText("");
		center.add(legends);

		JPanel footer = new JPanel();
		add(footer, BorderLayout.SOUTH);
		
		JButton ok = new JButton("OK");
		footer.add(ok);
		ok.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				
				returnViewOption = null;
				if (validateValues()) {
					returnViewOption = new GraphViewOption();
					
					String graphTitleText = graphTitle.getText().trim();
					returnViewOption.graphTitle = graphTitleText; 
					
					String legendText = legends.getText().trim();
					List<String> list = TextParserUtil.split(legendText, ",", null);
					if (list.size() > 1)
						returnViewOption.legends = list.toArray(new String[] { });
					else
						returnViewOption.legends = new String[] { };
					
				}
				dispose();
			}
		});
		
		
		JButton cancel = new JButton("Cancel");
		footer.add(cancel);
		cancel.addActionListener(new ActionListener() {
			
			@Override
			public void actionPerformed(ActionEvent e) {
				returnViewOption = null;
				dispose();
			}
		});

		
		setVisible(true);
	}
	
	
	/**
	 * Getting view option.
	 * @return view option.
	 */
	public GraphViewOption getViewOption() {
		return returnViewOption;
	}
	
	
	/**
	 * Validating values.
	 * @return true if values are validated.
	 */
	private boolean validateValues() {
		return true;
	}
	
	
}



/**
 * Declare a class that creates a window capable of being drawn to.
 * 
 * @author Michael T Flanagan
 * @version 1.0
 *
 */
class PlotGraphFlanagan extends Plot implements Serializable {

	
	/**
	 * Serial version UID for serializable class.
	 */
	protected static final long serialVersionUID = 1L; 

	
//	/**
//	 * Window frame for shared show method {@link #plot()}.
//	 */
//	JFrame window = new JFrame("Michael T Flanagan's plotting program - PlotGraph");

	/**
	 * Width of the window for the graph in pixels.
	 */
 	protected int graphWidth = 800;
 	
 	
 	/**
 	 * Height of the window for the graph in pixels.
 	 */
 	protected int graphHeight = 600;
 	
 	
 	/**
 	 * Choice 1: clicking on close icon causes window to close and the the program is exited.
 	 * Choice 2: clicking on close icon causes window to close and leaving the program running.
 	 */
 	protected int closeChoice = 1;
 	
 	
 	/**
 	 * Constructor with one 2-dimensional data arrays.
 	 * @param data one 2-dimensional data arrays.
 	 */
 	public PlotGraphFlanagan(double[][] data) {
 		super(data);
 	}


 	/**
 	 * Constructor with two 1-dimensional data arrays
 	 * @param xData x 1-dimensional data array.
 	 * @param yData y 1-dimensional data array.
 	 */
 	public PlotGraphFlanagan(double[] xData, double[] yData) {
 		super(xData, yData);
 	}
 	

 	/**
 	 * Re-scale the y dimension of the graph window and graph.
 	 * @param yScaleFactor y scale factor.
 	 */
 	public void rescaleY(double yScaleFactor) {
 		this.graphHeight = (int)Math.round((double)graphHeight*yScaleFactor);
 		super.yLen = (int)Math.round((double)super.yLen*yScaleFactor);
 		super.yTop = (int)Math.round((double)super.yTop*yScaleFactor);
 		super.yBot = super.yTop + super.yLen;
 	}

 	
 	/**
 	 * Re-scale the x dimension of the graph window and graph
 	 * @param xScaleFactor x scale factor.
 	 */
 	public void rescaleX(double xScaleFactor) {
     	this.graphWidth = (int)Math.round((double)graphWidth*xScaleFactor);
     	super.xLen = (int)Math.round((double)super.xLen*xScaleFactor);
     	super.xBot = (int)Math.round((double)super.xBot*xScaleFactor);
     	super.xTop = super.xBot + super.xLen;
 	}


 	/**
 	 * Get pixel width of the PlotGraph window.
 	 * @return pixel width of the PlotGraph window
 	 */
 	public int getGraphWidth() {
 		return this.graphWidth;
 	}

 	
 	/**
 	 * Get pixel height of the PlotGraph window.
 	 * @return pixel height of the PlotGraph window
 	 */
 	public int getGraphHeight() {
 		return this.graphHeight;
 	}

 	
 	/**
 	 * Reset height of graph window (pixels).
 	 * @param graphHeight height of graph window (pixels).
 	 */
 	public void setGraphHeight(int graphHeight) {
 		this.graphHeight = graphHeight;
 	}

 	
 	/**
 	 * Reset width of graph window (pixels).
 	 * @param graphWidth width of graph window (pixels).
 	 */
 	public void setGraphWidth(int graphWidth) {
 		this.graphWidth = graphWidth;
 	}

 	
 	/**
 	 * Get close choice.
 	 * @return close choice.
 	 */
 	public int getCloseChoice() {
 		return this.closeChoice;
 	}

 	
 	/**
 	 * Reset close choice.
 	 * @param choice close choice.
 	 */
 	public void setCloseChoice(int choice){
     	this.closeChoice = choice;
  	}


 	@Override
 	public void paint(Graphics g){
		// Rescale - needed for redrawing if graph window is resized by dragging
		double newGraphWidth = this.getSize().width;
		double newGraphHeight = this.getSize().height;
		double xScale = newGraphWidth/(double)this.graphWidth;
		double yScale = newGraphHeight/(double)this.graphHeight;
		rescaleX(xScale);
		rescaleY(yScale);
		
		// Call graphing method
		graph(g);
 	}

 	
 	/**
 	 * Plotting the graph. Set up the window and show graph.
 	 */
 	public void plot() {
 		plot2();
 		
//		// Set the initial size of the graph window
//		setSize(this.graphWidth, this.graphHeight);
//		
//		// Set background colour
//		window.getContentPane().setBackground(Color.white);
//		
//		// Choose close box
//		if(this.closeChoice == 1) {
//			window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//		}
//		else {
//			window.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
//		}
//		
//		// Add graph canvas
//		window.getContentPane().add("Center", this);
//		
//		// Set the window up
//		window.pack();
//		window.setResizable(true);
//		window.toFront();
//		
//		// Show the window
//		window.setVisible(true);
 	}

 	
 	/**
 	 * Plotting the graph. Set up the window and show graph.
 	 */
 	public void plot2() {
		JFrame window = new JFrame("Michael T Flanagan's plotting program - PlotGraph");
		
		setSize(this.graphWidth, this.graphHeight);
		
		// Set background color
		window.getContentPane().setBackground(Color.white);
		
		// Choose close box
		 window.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		
		// Add graph canvas
		window.getContentPane().add("Center", this);
		
		// Set the window up
		window.pack();
		window.setResizable(true);
		window.toFront();
		
		// Show the window
		window.setVisible(true);
 	}
 	
 	
 	/**
 	 * Displays dialogue box asking if you wish to exit program.
 	 * Answering yes end program - will simultaneously close the graph windows.
 	 */
 	public void endProgram(){
		int ans = JOptionPane.showConfirmDialog(null, "Do you wish to end the program\n"+"This will also close the graph window or windows", "End Program", JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE);
		if(ans == 0) {
			System.exit(0);
		}
		else {
			String message = "Now you must press the appropriate escape key/s, e.g. Ctrl C, to exit this program\n";
			if (this.closeChoice == 1) message += "or close a graph window";
			JOptionPane.showMessageDialog(null, message);
		}
 	}

 	
 	/**
 	 * Return the serial version unique identifier
 	 * @return the serial version unique identifier.
 	 */
 	public static long getSerialVersionUID() {
 		return PlotGraphFlanagan.serialVersionUID;
 	}

 	
}
