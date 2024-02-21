ifdef JAVA_HOME
	JAVAC ?= ${JAVA_HOME}/bin/javac
	JAVA ?= ${JAVA_HOME}/bin/java
	JAR ?= ${JAVA_HOME}/bin/jar
	NATIVE_IMAGE ?= ${JAVA_HOME}/bin/native-image
endif

JAVAC ?= javac
JAVA ?= java
JAR ?= jar
NATIVE_IMAGE ?= native-image

JAVA_COMPILE_OPTIONS = --enable-preview -source 21 -g --add-modules jdk.incubator.vector
JAVA_RUNTIME_OPTIONS += --enable-preview --add-modules jdk.incubator.vector
NATIVE_IMAGE_OPTIONS += --enable-preview --add-modules jdk.incubator.vector

JAVA_MAIN_CLASS = Llama2
JAR_FILE = llama2.jar

JAVA_SOURCES = $(wildcard *.java)
JAVA_CLASSES = $(patsubst %.java, target/classes/%.class, $(JAVA_SOURCES))

# Bundle all classes in a jar
$(JAR_FILE): $(JAVA_CLASSES) target/META-INF/MANIFEST.MF
	$(JAR) -cvfm $(JAR_FILE) target/META-INF/MANIFEST.MF -C target/classes .

jar: $(JAR_FILE)

# Compile the Java source files
compile: $(JAVA_CLASSES)
	$(info Java source files: $(JAVA_SOURCES))
	$(info Java .class files: $(JAVA_CLASSES))

# Prints the command to run the Java main class
run-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -cp target/classes $(JAVA_MAIN_CLASS)

# Prints the command to run the $(JAR_FILE)
run-jar-command:
	@echo $(JAVA) $(JAVA_RUNTIME_OPTIONS) -jar $(JAR_FILE)

# Clean the target directory
clean:
	rm -rf target
	rm $(JAR_FILE)
	rm default.iprof
	rm llama2

# Creates the manifest for the .jar file
target/META-INF/MANIFEST.MF:
	mkdir -p target/META-INF
	@echo "Manifest-Version: 1.0" > target/META-INF/MANIFEST.MF
	@echo "Class-Path: ." >> target/META-INF/MANIFEST.MF
	@echo "Main-Class: $(JAVA_MAIN_CLASS)" >> target/META-INF/MANIFEST.MF
	@echo "" >> target/META-INF/MANIFEST.MF

# Create a standalone executable of the llama2.jar using GraalVM
native-image: $(JAR_FILE)
	$(NATIVE_IMAGE) $(NATIVE_IMAGE_OPTIONS) -jar $(JAR_FILE)

# Compile the Java source files
target/classes/%.class: %.java
	$(JAVAC) $(JAVA_COMPILE_OPTIONS) -d target/classes $<

# Create the target directory
target/classes:
	mkdir -p target/classes

# Make the target directory a dependency of the Java class files
$(JAVA_CLASSES): target/classes
compile: target/classes
default: target/classes

.PHONY: compile clean jar run-command run-jar-command
.SUFFIXES: .java .class .jar .MF
