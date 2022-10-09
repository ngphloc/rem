/**
 * SIM: MACHINE LEARNING ALGORITHMS FRAMEWORK
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: sim.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.rem.regression.em.ui.graph;

import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.print.Printable;

/**
 * This interface represents a graph.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface Graph extends Printable {

	
	/**
	 * Setting up view option.
	 */
	void setupViewOption();
	
	
	/**
	 * Painting method.
	 * 
	 * @param g graphics context.
	 */
	void paint(Graphics g);
	
	
	/**
	 * Exporting image.
	 */
	void exportImage();
	
	
	/**
	 * Getting outer box.
	 * @return rectangle as outer box.
	 */
	Rectangle getOuterBox();
	
	
	/**
	 * Getting view option.
	 * @return view option.
	 */
	GraphViewOption getViewOption();
}


/**
 * This class represents the view option of a graph.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class GraphViewOption {
	
	
	/**
	 * Buffer ratio.
	 */
	public static final double BUFFER_RATIO = 0.1;

	
	/**
	 * Default tick option.
	 */
	public static final int TICK = 10;

	
	/**
	 * Default round option.
	 */
	public static final int ROUND = 2;
	
	
	/**
	 * Default decimal.
	 */
	protected static final int DECIMAL = 0; // Note number length = ROUND + DECIMAL

	
	/**
	 * X round option.
	 */
	public int xRound = ROUND;
	
	
	/**
	 * Y round option.
	 */
	public int yRound = ROUND;

	
	/**
	 * Z round option.
	 */
	public int zRound = ROUND;

	
	/**
	 * Round option for function.
	 */
	public int fRound = ROUND;

	
	/**
	 * Graph title.
	 */
	public String graphTitle = "";
	
	
	/**
	 * Legends
	 */
	public String[] legends = new String[0];

	
	/**
	 * Buffer ratio.
	 */
	public double bufferRatio = BUFFER_RATIO;

	
	/**
	 * Simplest mode.
	 */
	public boolean simplest = false;
	
	
	/**
	 * Default constructor.
	 */
	public GraphViewOption() {
		
	}
	
	
	/**
	 * Constructor with two round options.
	 * 
	 * @param xRound X round option.
	 * @param fRound Round option for function.
	 */
	public GraphViewOption(int xRound, int fRound) {
		this(xRound, ROUND, fRound);
	}
	
	
	/**
	 * Constructor with three round options.
	 * 
	 * @param xRound X round option.
	 * @param yRound Y round option.
	 * @param fRound Round option for function.
	 */
	public GraphViewOption(int xRound, int yRound, int fRound) {
		this(xRound, yRound, ROUND, fRound);
	}

	
	/**
	 * Constructor with all round options.
	 * 
	 * @param xRound X round option.
	 * @param yRound Y round option.
	 * @param zRound Z round option.
	 * @param fRound Round option for function.
	 */
	public GraphViewOption(int xRound, int yRound, int zRound, int fRound) {
		this.xRound = xRound;
		this.yRound = yRound;
		this.zRound = zRound;
		this.fRound = fRound;
	}
	
	
	/**
	 * Copying options from other object.
	 * @param that other view option.
	 */
	public void copy(GraphViewOption that) {
		this.xRound = that.xRound;
		this.yRound = that.yRound;
		this.zRound = that.zRound;
		this.fRound = that.fRound;
		
		this.graphTitle = that.graphTitle;
		if (this.graphTitle == null)
			this.graphTitle = "";
		
		this.legends = that.legends;
		if (this.legends == null)
			this.legends = new String[] { };
		
		this.bufferRatio = that.bufferRatio;
		this.simplest = that.simplest;
	}
	
	
}


/**
 * This class represents graph legend.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
class GraphLegendExp {
	
	
	/**
	 * Additional value.
	 */
	public double v = 1;
	
	
	/**
	 * Description text.
	 */
	public String text = "";
	
	
	/**
	 * Constructor with additional value and description text.
	 * @param v additional value
	 * @param text description text.
	 */
	private GraphLegendExp(double v, String text) {
		this.v = v;
		this.text = text;
	}
	
	
	/**
	 * Create a graph legend given number and round.
	 * @param number specified number.
	 * @param round specified round.
	 * @return a graph legend given number and round.
	 */
	public static GraphLegendExp create(double number, int round) {
		
		int d = (int) (number + 0.5);
		String dText = String.valueOf(d);
		int k = dText.length() - round;
		
		if (k == 0) {
			return new GraphLegendExp(1, "");
		}
		else if ( k > 0) {
			return new GraphLegendExp(
					1.0 / Math.pow(10, k),
					k == 1 ? " / 10" : " / 10^" + k
				);
		}
		else {
			k = Math.abs(k);
			return new GraphLegendExp(
					Math.pow(10, k),
					k == 1 ? " * 10" : " * 10^" + k
				);
		}
		
	}
	
	
}
