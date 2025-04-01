from tkinter import *
from tkinter import ttk
import numpy as np
import math

class Data:
    def __init__(self, nCols, nRows):
        self.cellClass = np.zeros(nCols*nRows, dtype="uint8")
        self.gridClass = ["Class 1", "Class 2"]
        self.nPoints = 0
        self.pointsLoc = []
        self.pointsClass = []
        self.pointClass = ["Class 1", "Class 2"]
        self.pointsRot = []
        self.pointsScale = []
        self.pointOriginalScalePxRad = 10
        self.nCols, self.nRows = nCols, nRows
        self.background = ""
        self.countSave = 0
    
    def reINIT(self, nCols, nRows):
        self.cellClass = np.zeros(nCols*nRows, dtype="uint8")
        self.gridClass = ["Class 1", "Class 2"]
        self.nPoints = 0
        self.pointsLoc = []
        self.pointsClass = []
        self.pointClass = ["Class 1", "Class 2"]
        self.pointsRot = []
        self.pointsScale = []
        self.pointOriginalScalePxRad = 10
        self.nCols, self.nRows = nCols, nRows
        self.background = ""

    def getCellClass(self): return self.cellClass
    def getPointsClass(self): return self.pointsClass
    def getPointsLoc(self): return self.pointsLoc
    def getPointsRot(self): return self.pointsRot
    def getPointsScale(self): return self.pointsScale
    def editCellClass(self, ind): 
        if self.cellClass[ind] == 0:
            self.cellClass[ind] = 1
        else:
            self.cellClass[ind] = 0
    def editPointsClass(self, ind): 
        if self.pointsClass[ind] == 0:
            self.pointsClass[ind] = 1
        else:
            self.pointsClass[ind] = 0    
    def addGridClass(self, gClass, classNo):
        self.gridClass[classNo] = gClass
    def addPointClass(self, pClass, classNo):
        self.pointClass[classNo] = pClass
    def addPointRot(self, x, y, classNo):
        self.pointsRot[classNo] = (x, y)
    def addPointScale(self, scale, classNo):
        self.pointsScale[classNo] = scale
    def addPoints(self, x, y):
        self.pointsLoc.append((x, y))
        self.pointsClass.append(0)
        self.pointsRot.append((x + 20, y))
        self.pointsScale.append(10)
        self.nPoints += 1
    
    def getGridCells(self):
        W = 600
        H = 600

        loc = []

        hCells = []
        wCells = []

        for i in range(int((H/self.nRows)/2), H, int(H/self.nRows)):
            hCells.append(i)
        for j in range(int((W/self.nCols)/2), W, int(W/self.nCols)):
            wCells.append(j)

        for i in hCells:
            for j in wCells:
                loc.append([j, 0, i])

        for i in range(self.nCols * self.nRows):
            loc[i][0] = loc[i][0] - int(W/2)
            loc[i][2] = loc[i][2] - int(H/2)
        
        return loc
    
    def magnitude(self, vector): 
        return math.sqrt(sum(pow(element, 2) for element in vector))
    
    def dotProd(self, u, v, N):
        prod = 0
        for i in range(N):
            prod = prod + u[i] * v[i]
        return prod
    
    def angleVector(self, u, v, N):
        dotProductOfVectors = self.dotProd(u, v, N)
        magOfu = self.magnitude(u)
        magOfv = self.magnitude(v)
 
        # angle between given vectors
        angle = (dotProductOfVectors
                / (magOfu * magOfv))
        
        return ((math.acos(angle) * 180)/math.pi)
 

    def save2npy(self):
        complete_dict = {}

        #Saving Floor Translations, Rotation, Scale and Class
        floor_1 = {}
        floor_2 = {}
        floorClass1Translation = []
        floorClass2Translation = []
        floorTranslations = self.getGridCells() #Saving Translations

        print(len(floorTranslations))
        print(len(self.gridClass))

        for i, translation in enumerate(floorTranslations):
            if self.cellClass[i] == 0: floorClass1Translation.append(translation)
            else: floorClass2Translation.append(translation)

        floor_1['t'] = floorClass1Translation
        floor_2['t'] = floorClass2Translation
        
        floor_1['r'] = [[0, 0, 0] for theta in range(len(floorClass1Translation))] #No Rotations
        floor_2['r'] = [[0, 0, 0] for theta in range(len(floorClass2Translation))] #No Rotations

        floor_1['s'] = [1 for s in range(len(floorClass1Translation))] #No Scaling
        floor_2['s'] = [1 for s in range(len(floorClass2Translation))] #No Scaling

        floor_1['c'] = "Photo of " + self.gridClass[0]
        floor_2['c'] = "Photo of " + self.gridClass[1]

        complete_dict['floor_1'] = floor_1
        complete_dict['floor_2'] = floor_2

        #Saving Object Translation
        object_1 = {}
        object_2 = {}

        objectClass1Translation = []
        objectClass2Translation = []
        objectClass1Rotation = []
        objectClass2Rotation = []
        objectClass1Scale = []
        objectClass2Scale = []

        for i, translation in enumerate(self.pointsLoc):
            rot = self.pointsRot[i]
            u = [rot[0] - translation[0], rot[1] - translation[1]]
            v = [20, 0]
            angle = self.angleVector(u, v, 2)
            if self.pointsClass[i] == 0: 
                objectClass1Translation.append([translation[0] - 300, 0, translation[1] - 300])
                objectClass1Rotation.append([0, angle, 0])   
            else: 
                objectClass2Translation.append([translation[0], 0, translation[1]])
                objectClass2Rotation.append([0, angle, 0])
        
        object_1['t'] = objectClass1Translation
        object_2['t'] = objectClass2Translation

        object_1['r'] = objectClass1Rotation
        object_2['r'] = objectClass2Rotation

        for i, scale in enumerate(self.pointsScale):
            if self.pointsClass[i] == 0:
                objectClass1Scale.append(scale/self.pointOriginalScalePxRad)
            else:
                objectClass2Scale.append(scale/self.pointOriginalScalePxRad)

        object_1['s'] = objectClass1Scale
        object_2['s'] = objectClass2Scale

        object_1['c'] = "Photo of " + self.pointClass[0]
        object_2['c'] = "Photo of " + self.pointClass[1]

        complete_dict["object_1"] = object_1
        complete_dict["object_2"] = object_2

        background_1 = {}
        background_1['t'] = [0,0,0]
        background_1['r'] = [0,0,0] 
        background_1['s'] = 1
        background_1['c'] = "Photo of " + self.background

        complete_dict["background_1"] = background_1

        np.save(str(self.countSave) + ".npy", complete_dict)
        self.countSave += 1

class GUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Instancing GUI")
        self.root.geometry("1200x600")
        self.nCols = 6
        self.nRows = 6
        self.mode = "INIT"
        self.startScale = FALSE
        self.scaleInd = 0
        self.MultiSelect = FALSE
        self.MultiSelectPoints = []
        self.data = Data(self.nCols, self.nRows)

        #Initializing Widgets
        self.canvas = Canvas(self.root, height=600, width=600, bg="white")
        self.gLabel = Label(self.root, text="Grid - Draw Grid")
        self.nColsLabel = Label(self.root, text="# of Cols:")
        self.nColsEntry = Entry(self.root, width = 10)
        self.nRowsLabel = Label(self.root, text="# of Rows:")
        self.nRowsEntry = Entry(self.root, width = 10)
        self.gridButton = Button(self.root, text="Enter Grid Info", command=self.enterGridStats)
        self.btnCells = Button(self.root, text="Grid and Cells", command=self.gridMode)
        self.btnPoints = Button(self.root, text="Add Points", command=self.pointsMode)
        #Grid Class Entry
        self.nGridClassOneLabel = Label(self.root, text="Class 1:")
        self.nGridClassOneEntry = Entry(self.root, width = 10)
        self.nGridClassTwoLabel = Label(self.root, text="Class 2:")
        self.nGridClassTwoEntry = Entry(self.root, width = 10)
        self.gridClassButton = Button(self.root, text="Enter Grid Classes", command=self.enterGridClass)
        self.exitGrids = Button(self.root, text="Exit Grid Mode", command=self.exit)
        #Point Class Entry
        self.pointClassOneLabel = Label(self.root, text="Point Class 1:")
        self.pointClassOneEntry = Entry(self.root, width = 10)
        self.pointClassTwoLabel = Label(self.root, text="Point Class 2:")
        self.pointClassTwoEntry = Entry(self.root, width = 10)
        self.btnPointsClass = Button(self.root, text="Add Points Class", command=self.enterPointClass)
        #Point Orientation Enabling
        self.btnPointsOrientation = Button(self.root, text="Set Orientation", command=self.enableOrientation)
        #Handling Point Scale
        self.btnPointsScale = Button(self.root, text="Set Scale", command=self.enablePointScale)
        #Background and Saving
        self.backgroundLabel = Label(self.root, text="Add Background:")
        self.backgroundEntry = Entry(self.root, width = 10)
        self.backgroundEntryBtn = Button(self.root, text="Enter Background", command=self.enterBackground)
        self.saveData = Button(self.root, text="Save Data", command=self.data.save2npy)
        #Multi Select
        self.multiSelectBtn = Button(self.root, text="Enable Multi Select Point", command=self.enableMultiSelect)


        #Packing
        self.canvas.grid(column=1, row=1, rowspan=11)
        self.gLabel.grid(column=2, row=1)
        self.btnCells.grid(column=3, row=2)
        self.btnPoints.grid(column=5, row=2)
        self.nColsLabel.grid(column=2, row=3)
        self.nColsEntry.grid(column=3, row=3)
        self.nRowsLabel.grid(column=2, row=4)
        self.nRowsEntry.grid(column=3, row=4)
        self.gridButton.grid(column=3, row=5)
        #Packing Grid Class Entry
        self.nGridClassOneLabel.grid(column=2, row=6)
        self.nGridClassOneEntry.grid(column=3, row=6)
        self.nGridClassTwoLabel.grid(column=2, row=7)
        self.nGridClassTwoEntry.grid(column=3, row=7)
        self.gridClassButton.grid(column=3, row=8)
        self.exitGrids.grid(column=3, row=9)
        #Packing Point Class Entry
        self.pointClassOneLabel.grid(column=4, row=3)
        self.pointClassOneEntry.grid(column=5, row=3)
        self.pointClassTwoLabel.grid(column=4, row=4)
        self.pointClassTwoEntry.grid(column=5, row=4)
        self.btnPointsClass.grid(column=5, row=5)
        #Packing Point Orientation
        self.btnPointsOrientation.grid(column=5, row=7)
        #Packing Point Scale
        self.btnPointsScale.grid(column=5, row=8)
        #Packing Background and Saving
        self.backgroundLabel.grid(column=4, row = 9)
        self.backgroundEntry.grid(column=5, row=9)
        self.backgroundEntryBtn.grid(column=5, row=10)
        self.saveData.grid(column=5, row=11)
        #Multi select
        self.multiSelectBtn.grid(column=5, row=6)

    def enableMultiSelect(self): 
        self.MultiSelect = TRUE
        self.gLabel.config(text="Point Multi Select Mode Enabled")
    
    def getOrigin(self, eventOrigin):
        global x,y
        x = eventOrigin.x
        y = eventOrigin.y
        if self.mode == "GRID CELL CLASS":
            w, h = 600, 600
            xMov, yMov = int(w/self.nCols), int(h/self.nRows)
            nColsTemp, nRowsTemp = self.nCols, self.nRows

            while nRowsTemp > 0:
                if y >= (h - yMov):
                    break
                else:
                    nRowsTemp -= 1
                h = h - yMov

            while nColsTemp > 0:
                if x >= (w - xMov):
                    break
                else:
                    nColsTemp -= 1
                w = w - xMov

            ind = self.nCols * (nRowsTemp - 1) + nColsTemp - 1
            self.data.editCellClass(ind)

            self.createGrid()
            self.displayGridClass()

        elif self.mode == "POINT MODE":
            self.data.addPoints(x, y)
            self.displayPoint(x, y)
        
        elif self.mode == "POINT CLASS":
            if self.MultiSelect == FALSE:
                min_dist = 1e9
                global min_dist_ind
                loc = self.data.getPointsLoc()
                for i, point in enumerate(loc):
                    nDist = math.dist([x, y], [point[0], point[1]])
                    if nDist < min_dist: 
                        min_dist = nDist
                        min_dist_ind = i
                self.data.editPointsClass(min_dist_ind)
                self.createGrid()
                self.displayGridClass()
                self.displayPointClass()
            else:
                self.MultiSelectPoints.append([x, y])
                if len(self.MultiSelectPoints) == 2:
                    boundary = self.MultiSelectPoints
                    loc = self.data.getPointsLoc()
                    indices = []
                    for i, point in enumerate(loc):
                        if point[0] >= boundary[0][0] and point[0] <= boundary[1][0] and point[1] >= boundary[0][1] and point[1] <= boundary[1][1]:
                            indices.append(i)
                    
                    for i in indices:
                        self.data.editPointsClass(i)
                    
                    self.createGrid()
                    self.displayGridClass()
                    self.displayPointClass()

                    self.MultiSelectPoints = []
                    self.MultiSelect = FALSE
        
        elif self.mode == "POINT ORIENTATION":
            min_dist = 1e9
            min_dist_ind
            loc = self.data.getPointsLoc()
            for i, point in enumerate(loc):
                nDist = math.dist([x, y], [point[0], point[1]])
                if nDist < min_dist: 
                    min_dist = nDist
                    min_dist_ind = i
            #cRotation = self.data.getPointsRot()
            self.data.addPointRot(x, y, min_dist_ind)
            self.createGrid()
            self.displayGridClass()
            self.displayPointClass()
            self.genVectors()
        
        elif self.mode == "POINT SCALING":
            if self.startScale == FALSE:
                min_dist = 1e9
                min_dist_ind
                loc = self.data.getPointsLoc()
                for i, point in enumerate(loc):
                    nDist = math.dist([x, y], [point[0], point[1]])
                    if nDist < min_dist: 
                        min_dist = nDist
                        min_dist_ind = i
                self.scaleInd = min_dist_ind
                self.startScale = TRUE
            elif self.startScale == TRUE:
                self.startScale = FALSE
        print(x, y)
    
    def motion(self, event):
        x, y = event.x, event.y
        if self.startScale == TRUE:
            locs = self.data.getPointsLoc()
            loc = locs[self.scaleInd]
            rad = math.dist([loc[0], loc[1]], [x, y])
            self.data.addPointScale(2 * rad, self.scaleInd)
            self.createGrid()
            self.displayGridClass()
            self.displayPointClass()
            self.displayPointScale()
            self.genVectors()
            print('{}, {}'.format(x, y))

    def start(self):
        #self.root.bind("<Button 1>", getOrigin)
        self.canvas.bind("<Button 1>", self.getOrigin)
        self.canvas.bind('<Motion>', self.motion)
        self.root.mainloop()

    def createGrid(self, event=None):
        w = 600
        h = 600

        self.canvas.delete('all')

        #Creating Vertical Line
        for i in range(0, w, int(w/self.nCols)):
            self.canvas.create_line([(i, 0), (i, h)], tag='grid_line')
        
        #Creating Horizontal Line
        for i in range(0, h, int(h/self.nRows)):
            self.canvas.create_line([(0, i), (w, i)], tag="grid_line")

    def enterGridStats(self):
        self.nCols = int(self.nColsEntry.get())
        self.nRows = int(self.nRowsEntry.get())
        self.createGrid()
        self.data.reINIT(self.nCols, self.nRows)
        self.gLabel.config(text="Grid - Enter Cell Classes")

    def gridMode(self):
        self.gLabel.config(text="Grid - Draw Grid")

    def pointsMode(self):
        self.gLabel.config(text="Point Placement")
        self.mode = "POINT MODE"

    def exit(self):
        self.gLabel.config(text="No Curr Mode")
        self.mode = "INIT"

    def displayGridClass(self):
        w = 600
        h = 600

        self.canvas.delete('grid_class&&token')

        cellClassLoc = []

        hCells = []
        wCells = []
        for i in range(int((h/self.nRows)/2), h, int(h/self.nRows)):
            hCells.append(i)
        for j in range(int((w/self.nCols)/2), w, int(w/self.nCols)):
            wCells.append(j)

        for i in hCells:
            for j in wCells:
                cellClassLoc.append((j, i))

        cell_class = self.data.getCellClass()

        for i in range(self.nCols * self.nRows):
            self.canvas.create_text(cellClassLoc[i], text=str(cell_class[i] + 1), font="Calibri 20 bold", fill="light gray", tag="grid_class")

    def displayPoint(self, x, y):
        self.canvas.create_oval(x-5, y-5, x+5, y+5, width = 0, fill = 'red')

    def displayPointClass(self):
        loc = self.data.getPointsLoc()
        cClass = self.data.getPointsClass()
        
        for i, l in enumerate(loc):
            color = ['red', 'blue']
            self.canvas.create_oval(l[0] - 5, l[1] - 5, l[0] + 5, l[1] + 5, width = 0, fill = color[cClass[i]])

    def displayPointScale(self):
        loc = self.data.getPointsLoc()
        scale = self.data.getPointsScale()
        cClass = self.data.getPointsClass()
        
        for i, l in enumerate(loc):
            s = scale[i]
            color = ['red', 'blue']
            self.canvas.create_oval(l[0] - int(s/2), l[1] - int(s/2), l[0] + int(s/2), l[1] + int(s/2), width = 0, fill = color[cClass[i]])

    def enterGridClass(self):
        self.data.addGridClass(self.nGridClassOneEntry.get(), 0)
        self.data.addGridClass(self.nGridClassTwoEntry.get(), 1)
        self.mode = "GRID CELL CLASS"
        self.gLabel.config(text="Grid - Click on Cell to change class")
        self.displayGridClass()

    def enterPointClass(self):
        self.data.addPointClass(self.pointClassOneEntry.get(), 0)
        self.data.addPointClass(self.pointClassTwoEntry.get(), 1)
        self.mode = "POINT CLASS"
        self.gLabel.config(text="Point - Click on Point to change class")

    def enableOrientation(self):
        self.genVectors()
        self.mode = "POINT ORIENTATION"
        self.gLabel.config(text="Point - Change Button Orientation")

    def genVectors(self):
        loc = self.data.getPointsLoc()
        rot = self.data.getPointsRot()
        
        for i, l in enumerate(loc):
            self.canvas.create_line(l, rot[i], fill="black")

    def enablePointScale(self):
        self.mode = "POINT SCALING"
        self.gLabel.config(text="Point - Point Scale")

    def enterBackground(self):
        self.data.background = self.backgroundEntry.get()
appStart = GUI()
appStart.start()