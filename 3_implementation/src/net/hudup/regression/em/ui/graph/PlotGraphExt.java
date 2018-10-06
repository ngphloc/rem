package net.hudup.regression.em.ui.graph;

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
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JTextField;

import flanagan.plot.PlotGraph;
import net.hudup.core.Util;
import net.hudup.core.logistic.UriAssoc;
import net.hudup.core.logistic.xURI;
import net.hudup.core.logistic.ui.UIUtil;
import net.hudup.core.parser.TextParserUtil;

/**
 * This class is an extension of plot graph.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class PlotGraphExt extends PlotGraph implements Graph {

	
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
		// TODO Auto-generated constructor stub
	}

	
	/**
	 * Constructor with data array.
	 * @param data specified data array.
	 */
	public PlotGraphExt(double[][] data) {
		super(data);
		// TODO Auto-generated constructor stub
	}

	
	@Override
	public int print(Graphics g, PageFormat pf, int pageIndex)
			throws PrinterException {
		// TODO Auto-generated method stub
		
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
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
	}

	
	@Override
	public Rectangle getOuterBox() {
		// TODO Auto-generated method stub
		return new Rectangle(0, 0, graphWidth, graphHeight);
	}


	@Override
	public void setupViewOption() {
		// TODO Auto-generated method stub
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
		super (UIUtil.getFrameForComponent(comp), "Plot graph option", true);
		
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		setSize(400, 200);
		setLocationRelativeTo(UIUtil.getFrameForComponent(comp));
		
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
				// TODO Auto-generated method stub
				
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
				// TODO Auto-generated method stub
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
