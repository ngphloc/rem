package net.rem.regression.ui.graph;

/**
 * This class is another extension of plot graph.
 * 
 * @author Loc Nguyen, Michael T Flanagan
 * @version 1.0
 *
 */
public class PlotGraphExt2 extends PlotGraphExt {
	
	
	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Graph feature.
	 */
	protected String graphFeature = "No feature";
	
	
	/**
	 * Constructor with X data, Y data, and graph feature.
	 * @param xData specified X data.
	 * @param yData specified Y data.
	 * @param graphFeature graph feature.
	 */
	public PlotGraphExt2(double[] xData, double[] yData, String graphFeature) {
		super(xData, yData);
		if (graphFeature != null) this.graphFeature = graphFeature; 
	}

	
	/**
	 * Constructor with data array and graph feature..
	 * @param data specified data array.
	 * @param graphFeature graph feature.
	 */
	public PlotGraphExt2(double[][] data, String graphFeature) {
		super(data);
		if (graphFeature != null) this.graphFeature = graphFeature; 
	}

	
	@Override
	public String getGraphFeature() {
		return graphFeature;
	}


}

	
