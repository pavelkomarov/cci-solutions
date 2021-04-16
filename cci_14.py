# https://realpython.com/python-sql-libraries/
# https://docs.python.org/3/library/sqlite3.html
 
import sqlite3 # serverless and self-contained! Hooray! Comes included in python, writes to a file.

# set up a connection
connection = sqlite3.connect(":memory:") # This special string means the database will be in memory
cursor = connection.cursor()
# print(cursor.execute("PRAGMA foreign_keys;").fetchone()) prints 0, so no foreign key support

# Create tables
queries = []
queries.append("""
CREATE TABLE IF NOT EXISTS Apartments (
	AptID INTEGER PRIMARY KEY AUTOINCREMENT,
	UnitNumber VARCHAR(10),
	BuildingID INTEGER REFERENCES Buildings(BuildingID)
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS Buildings (
	BuildingID INTEGER PRIMARY KEY AUTOINCREMENT,
	ComplexID INTEGER REFERENCES Complexes(ComplexID),
	BuildingName VARCHAR(100),
	Address VARCHAR(500)
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS Requests (
	RequestID INTEGER PRIMARY KEY AUTOINCREMENT,
	Status VARCHAR(100),
	AptID INTEGER REFERENCES Apartments(AptID),
	Description VARCHAR(500)
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS Complexes (
	ComplexID INTEGER PRIMARY KEY AUTOINCREMENT,
	ComplexName VARCHAR(100)
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS Tenants (
	TenantID INTEGER PRIMARY KEY AUTOINCREMENT,
	TenantName VARCHAR(100)
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS AptTenants (
	TenantID REFERENCES Tenants(TenantID),
	AptID REFERENCES Apartments(AptID)
);
""")
for q in queries: cursor.execute(q)
connection.commit() # all queries take effect in database

# Fill tables
queries = []
queries.append("""
INSERT INTO
	Complexes (ComplexName)
VALUES
	('City Island'),
	('Pierpont by Urbana'),
	('Racquet Club Villas'),
	('The Bowie');
""")
queries.append("""
INSERT INTO
	Buildings (ComplexID, BuildingName, Address)
VALUES
	(1, "Babi and Joe's Building", '30 Pilot St'),
	(2, "Pavel's building", '315 Pierpont Ave'),
	(3, "The Florida House", '6003 Courtside Dr'),
	(4, "The Bowie Tower", '311 Bowie St');
""")
queries.append("""
INSERT INTO
	Apartments (UnitNumber, BuildingID)
VALUES
	('7M', 1),
	('6B', 1),
	('706', 2),
	('613', 2),
	("Chris'Room", 3),
	('1308', 4);
""")
queries.append("""
INSERT INTO
	Tenants (TenantName)
VALUES
	('Babi'),
	('Joe'),
	('Pavel'),
	('McKay'),
	('Chris'),
	('Eion');
""")
queries.append("""
INSERT INTO
	AptTenants (TenantID, AptID)
VALUES
	(1, 2),
	(2, 1),
	(3, 3),
	(4, 4),
	(5, 5),
	(6, 5),
	(5, 6);
""")
queries.append("""
INSERT INTO
	Requests (Status, AptID, Description)
VALUES
	('Closed', 3, 'Pavel still lives there.'),
	('Accepted', 6, 'charming chap.'),
	('Open', 1, ''),
	('Open', 2, ''),
	('Open', 4, '');
""")
for q in queries: cursor.execute(q)
connection.commit()

# Useful code to look at a whole little table
# cursor.execute("SELECT * FROM Tenants")
# for x in cursor.fetchall(): print(x)

# One
# First find a subtable of all TenantIDs with more than one entry in the AptTenants table.
# Then join that table to the Tenants table to get their names.
one = """
SELECT
	TenantName
FROM
	Tenants INNER JOIN
		(SELECT TenantID FROM AptTenants
		GROUP BY TenantID HAVING COUNT(*) > 1) T
ON Tenants.TenantID = T.TenantID
"""
cursor.execute(one)
assert cursor.fetchall() == [('Chris',)]

# Two
# First join the Requests and Apartments tables together, accepting only entries where the
# request is open. Group by building so we can sum up how many such requests there are per
# building. Then left join that result to the Buildings table to get the building names.
# The left join inserts null values where the left table has no corresponding thing from
# the right table. We know in this case that means there are zero open requests for that
# building, so insert zero in those cases instead.
two = """
SELECT
	BuildingName, IFNULL(Cnt, 0) as Cnt
FROM
	Buildings LEFT JOIN
		(SELECT BuildingID, COUNT(*) as Cnt
		FROM Requests INNER JOIN Apartments
		ON Requests.AptID = Apartments.AptID
		WHERE Requests.Status = 'Open'
		GROUP BY BuildingID) T
	ON Buildings.BuildingID = T.BuildingID
"""
cursor.execute(two)
assert cursor.fetchall() == [("Babi and Joe's Building", 2),
							 ("Pavel's building", 1),
							 ('The Florida House', 0),
							 ('The Bowie Tower', 0)]

# Three
# I'm closing requests for building #1 instead
three = """
UPDATE
	Requests
SET
	Status = "Closed"
WHERE
	AptID IN (SELECT AptID FROM Apartments WHERE BuildingID = 1)
"""
cursor.execute(three)
cursor.execute("SELECT * FROM Requests")
assert cursor.fetchall() == [(1, 'Closed', 3, 'Pavel still lives there.'),
							 (2, 'Accepted', 6, 'charming chap.'),
							 (3, 'Closed', 1, ''),
							 (4, 'Closed', 2, ''),
							 (5, 'Open', 4, '')]
connection.close() # We're done with this database, so close it, which destroys it, since it's in memory.

# Four
# Say I have [(1, a),  and [(1, z),
#			  (2, b),		(2, y),
#			  (3, c),		(3, x),
#			  (5, e)]		(4, w)]
# If I outer join on the numerical column, then I'll end up with a table with rows 1 through 5, where
# e and w have no companions in oposite columns. If I inner join, then I'll end up with columns 1-3,
# where e and w will be entirely missing specifically because information for those rows can't be made
# complete. If I left join, then whichever is the first table in the "multiplication" will have all its
# rows represented, with information added in as possible from the right table, here meaning rows 1,2,3,5,
# so w goes missing. And if I right join, then the right table gets this preference, so the result has
# rows 1,2,3,4, and e goes missing. In all cases, if a row doesn't have full data, the missing locations
# are filled with null, or None in case of the python reading. Which join is right for your situation
# depends on which rows you want your result to have. In Two, for instance, we wanted *all* the buildings
# to be in the output, not just the ones with open requests, so we needed to left join to our open
# requests temporary-table and fill the nulls with 0. Had we inner joined, the buildings with no open
# requests would simply be omitted from the result.

# Five
# Denormalization is a new phrase to me, but I gather that it means keeping columns in multiple different
# tables, so that reads can happen faster, at the cost of storage space. It's a classic space-for-compute
# tradeoff. If we were to be most space efficient and duplicate no information across tables, then we
# end up in a situation like the first few questions above, where doing anything even relatively minor
# involves numerous nested, complicated joins. As the tables grow larger, accomplishing these joins will
# grow more and more expensive and time consuming, so better to anticipate which pieces of information
# will be needed together and store them together ahead of time, almost like a sort of caching, so
# selection and can be made much less knotty. A downside is that we may have to update information in
# several tables if something changes, and this of course increases the likelihood we miss something.

# Six
# Apparently "entity relationship diagram" is just meant to be a bunch of boxes showing what things are,
# what attributes they have, and what relationships they have. Here we've got that a Professional is a
# Person, one kind of relationship, and that a Professional works for a company, another kind of
# relationship. I guess this is just meant to be an initial database design step, to give a picture of
# what's involved.
#
#	Person: (ID, Name, Birth Date, Phone, etc.)
#		^
#		| is a
#	Professional: (Degree, Experience)
# 		|
#		| N
#	[Works For: (Salary, Start Date)]
#		|
#		| 1
#	Company: (ID, Name, Address, Industry, etc.)

# Seven
# I'm assuming all courses are weighted the same
connection = sqlite3.connect(":memory:") # new database
cursor = connection.cursor()

# Create
queries = []
queries.append("""
CREATE TABLE IF NOT EXISTS Students (
	ID INTEGER PRIMARY KEY AUTOINCREMENT,
	Name VARCHAR(100),
	Year INTEGER
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS Courses (
	ID INTEGER PRIMARY KEY AUTOINCREMENT,
	Name VARCHAR(100),
	Department VARCHAR(100)
);
""")
queries.append("""
CREATE TABLE IF NOT EXISTS Grades (
	StudentID REFERENCES Students(ID),
	CourseID REFERENCES Courses(ID),
	Outcome INTEGER
);
""")
for q in queries: cursor.execute(q)
connection.commit()

# Fill
queries = []
queries.append("""
INSERT INTO
	Students (Name, Year)
VALUES
	('Andrew', 2020),
	('Elizabeth', 2021),
	('John', 2020),
	('Fabio', 2022);
""")
queries.append("""
INSERT INTO
	Courses (Name, Department)
VALUES
	('Biochemistry', 'Chemistry'),
	('Signal Processsing', 'Electrical Engineering'),
	('Thermodynamics', 'Mechanical Engineering'),
	('Orbital Mechanics', 'Aerospace Engineering'),
	('Relativity', 'Physics');
""")
queries.append("""
INSERT INTO
	Grades (StudentID, CourseID, Outcome)
VALUES
	(1, 1, 3),
	(1, 2, 4),
	(2, 1, 4),
	(2, 4, 4),
	(3, 5, 2),
	(3, 3, 3),
	(4, 5, 4),
	(4, 4, 3);
""")
for q in queries: cursor.execute(q)
connection.commit()

# First off, the solution relies on a variable, but sqlite doesn't have variables,
# so I have to create a temp table
# https://stackoverflow.com/questions/7739444/declare-variable-in-sqlite-and-use-it
seven = []
seven.append("""
CREATE TEMP TABLE _Variables (Name TEXT PRIMARY KEY, v REAL);
""")
seven.append("""
INSERT INTO _Variables (Name) VALUES('GPACutoff')
""")
# Next, sqlite doesn't have the SELECT TOP X PERCENT function, so I have to do something else.
# In my innermost selection here, I'm finding average GPA per student. Then I calculate the
# percent rank of each student based on their GPA in a new table. Then I join that table
# to the Students, so I can get their names, tossing all rows with a percent rank < 0.9.
seven.append("""
SELECT Name, GPA
FROM
	Students INNER JOIN
	(SELECT StudentID, GPA, PERCENT_RANK() OVER(ORDER BY GPA) as Rank
		FROM (
			SELECT AVG(Outcome) as GPA, StudentID
			FROM Grades
			GROUP BY Grades.StudentID
		)
	) T
	ON Students.ID = T.StudentID
	WHERE Rank >= 0.9
""")
for q in seven: cursor.execute(q)
assert cursor.fetchall() == [('Elizabeth', 4.0)]
