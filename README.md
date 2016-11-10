# attend
A top-down visual attentional model for the Soar cognitive architecture [1]. Model integrates an adapted openCV implementation of the bottom-up visual saliency model in [2] with parametrized normalization for integrating an agent's top-down knowledge to create object specific saliency maps. The implemetation is inteded to be used as a ranking metric for object proposals to improve visual detection and identification. edgeBox [3] will be used for object proposal generation. 

Current Features used:
	- Color
	- Color Opponency
	- Orientation
	- Intensity

[1] Laird, John E. The Soar cognitive architecture. MIT Press, 2012.
[2] Itti, Laurent, Christof Koch, and Ernst Niebur. "A model of saliency-based visual attention for rapid scene analysis." IEEE Transactions on pattern analysis and machine intelligence 20.11 (1998): 1254-1259.
[3] Zitnick, C. Lawrence, and Piotr Dollár. "Edge boxes: Locating object proposals from edges." European Conference on Computer Vision. Springer International Publishing, 2014.

