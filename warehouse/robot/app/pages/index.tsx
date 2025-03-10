"use client";
import { NextPage } from "next";
import React, { useEffect, useRef, useState, forwardRef, memo, useImperativeHandle } from "react";
import Head from "next/head";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { Physics, usePlane, useBox } from "@react-three/cannon";
import * as THREE from "three";

interface RobotParts {
  base?: THREE.Object3D;
  shoulder?: THREE.Object3D;
  upperArm?: THREE.Object3D;
  elbow?: THREE.Object3D;
  forearm?: THREE.Object3D;
  pincerBase?: THREE.Object3D;
  pincerClaw1?: THREE.Object3D;
  pincerClaw2?: THREE.Object3D;
  camera?: THREE.PerspectiveCamera;
}

class RobotController {
  robotParts: RobotParts;
  demoMode = false;
  commandQueue: { part: string; target: number }[] = [];
  isMoving = false;
  ROT_SPEED = THREE.MathUtils.degToRad(150) / 60;
  LERP_FACTOR = 0.1;
  shoulderMin = THREE.MathUtils.degToRad(0);
  shoulderMax = THREE.MathUtils.degToRad(45);
  elbowMin = THREE.MathUtils.degToRad(-90);
  elbowMax = THREE.MathUtils.degToRad(45);
  constructor(robotParts: RobotParts) {
    this.robotParts = robotParts;
  }
  rotateBase(direction: number) {
    if (!this.robotParts.base) return;
    this.robotParts.base.rotation.y += this.ROT_SPEED * direction;
  }
  moveShoulder(direction: number) {
    if (!this.robotParts.shoulder) return;
    const delta = this.ROT_SPEED * direction;
    this.robotParts.shoulder.rotation.x = THREE.MathUtils.clamp(
      this.robotParts.shoulder.rotation.x + delta,
      this.shoulderMin,
      this.shoulderMax
    );
  }
  moveElbow(direction: number) {
    if (!this.robotParts.elbow) return;
    const delta = this.ROT_SPEED * direction;
    this.robotParts.elbow.rotation.x = THREE.MathUtils.clamp(
      this.robotParts.elbow.rotation.x + delta,
      this.elbowMin,
      this.elbowMax
    );
  }
  resetToStart() {
    if (this.robotParts.base) this.robotParts.base.rotation.y = 0;
    if (this.robotParts.shoulder) this.robotParts.shoulder.rotation.x = 0;
    if (this.robotParts.elbow)
      this.robotParts.elbow.rotation.x = THREE.MathUtils.degToRad(45);
    this.commandQueue = [];
    this.isMoving = false;
  }
  addCommand(part: string, deltaRad: number) {
    const { base, shoulder, elbow } = this.robotParts;
    let target = 0;
    switch (part) {
      case "B":
        if (base) target = base.rotation.y + deltaRad;
        break;
      case "S":
        if (shoulder) {
          target = THREE.MathUtils.clamp(
            shoulder.rotation.x + deltaRad,
            this.shoulderMin,
            this.shoulderMax
          );
        }
        break;
      case "E":
        if (elbow) {
          target = THREE.MathUtils.clamp(
            elbow.rotation.x + deltaRad,
            this.elbowMin,
            this.elbowMax
          );
        }
        break;
    }
    this.commandQueue.push({ part, target });
  }
  update() {
    const { base, shoulder, elbow } = this.robotParts;
    if (this.demoMode) {
      this.rotateBase(1);
      return;
    }
    if (!this.isMoving && this.commandQueue.length > 0) {
      this.isMoving = true;
    }
    if (this.isMoving && this.commandQueue.length > 0) {
      const { part, target } = this.commandQueue[0];
      let currentRotation = 0;
      let partRef: THREE.Object3D | undefined;
      switch (part) {
        case "B":
          partRef = base;
          currentRotation = base?.rotation.y || 0;
          break;
        case "S":
          partRef = shoulder;
          currentRotation = shoulder?.rotation.x || 0;
          break;
        case "E":
          partRef = elbow;
          currentRotation = elbow?.rotation.x || 0;
          break;
      }
      if (partRef) {
        const delta = (target - currentRotation) * this.LERP_FACTOR;
        const newRotation = currentRotation + delta;
        if (part === "B") partRef.rotation.y = newRotation;
        else partRef.rotation.x = newRotation;
        if (Math.abs(newRotation - target) < 0.01) {
          this.commandQueue.shift();
          this.isMoving = false;
        }
      }
    }
  }
}

interface RobotArmProps {
  position?: [number, number, number];
}

const RobotArm = memo(
  forwardRef<RobotParts, RobotArmProps>((props, ref) => {
    const { position = [0, 0.1, 0] } = props;
    const baseRef = useRef<THREE.Group>(null);
    const shoulderRef = useRef<THREE.Group>(null);
    const upperArmRef = useRef<THREE.Group>(null);
    const elbowRef = useRef<THREE.Group>(null);
    const forearmRef = useRef<THREE.Group>(null);
    const pincerBaseRef = useRef<THREE.Group>(null);
    const pincerClaw1Ref = useRef<THREE.Mesh>(null);
    const pincerClaw2Ref = useRef<THREE.Mesh>(null);
    const forearmCameraRef = useRef<THREE.PerspectiveCamera>(null);
    useImperativeHandle(ref, () => ({
      base: baseRef.current!,
      shoulder: shoulderRef.current!,
      upperArm: upperArmRef.current!,
      elbow: elbowRef.current!,
      forearm: forearmRef.current!,
      pincerBase: pincerBaseRef.current!,
      pincerClaw1: pincerClaw1Ref.current!,
      pincerClaw2: pincerClaw2Ref.current!,
      camera: forearmCameraRef.current!,
    }));
    return (
      <group position={position}>
        <group ref={baseRef}>
          <mesh castShadow>
            <cylinderGeometry args={[0.2, 0.2, 0.2, 32]} />
            <meshNormalMaterial />
          </mesh>
          <group ref={shoulderRef} position={[0, 0.1, 0]}>
            <mesh castShadow>
              <sphereGeometry args={[0.2, 32, 32]} />
              <meshNormalMaterial />
            </mesh>
            <group ref={upperArmRef} position={[0, 0.5, 0]}>
              <mesh castShadow>
                <boxGeometry args={[0.1, 1, 0.1]} />
                <meshNormalMaterial />
              </mesh>
              <group
                ref={elbowRef}
                position={[0, 0.6, 0]}
                rotation={[THREE.MathUtils.degToRad(45), 0, Math.PI / 2]}
              >
                <mesh castShadow>
                  <cylinderGeometry args={[0.15, 0.15, 0.15, 32]} />
                  <meshNormalMaterial />
                </mesh>
                <group ref={forearmRef} position={[0, 0, 0.4]}>
                  <mesh castShadow>
                    <boxGeometry args={[0.1, 0.1, 0.7]} />
                    <meshNormalMaterial />
                  </mesh>
                  <group ref={pincerBaseRef} position={[0, 0, 0.36]}>
                    <mesh castShadow>
                      <boxGeometry args={[0.2, 0.4, 0.05]} />
                      <meshNormalMaterial />
                    </mesh>
                    <mesh ref={pincerClaw1Ref} position={[0, 0.175, 0.075]} castShadow>
                      <boxGeometry args={[0.2, 0.05, 0.1]} />
                      <meshNormalMaterial />
                    </mesh>
                    <mesh ref={pincerClaw2Ref} position={[0, -0.175, 0.075]} castShadow>
                      <boxGeometry args={[0.2, 0.05, 0.1]} />
                      <meshNormalMaterial />
                    </mesh>
                  </group>
                  <perspectiveCamera
                    ref={forearmCameraRef}
                    fov={60}
                    aspect={16 / 9}
                    near={0.1}
                    far={100}
                    position={[0, 0, 0.45]}
                    rotation={[
                      THREE.MathUtils.degToRad(0),
                      THREE.MathUtils.degToRad(180),
                      THREE.MathUtils.degToRad(90),
                    ]}
                  />
                </group>
              </group>
            </group>
          </group>
        </group>
      </group>
    );
  })
);

RobotArm.displayName = "RobotArm";

const Floor: React.FC<{ position?: [number, number, number] }> = ({ position }) => {
  const [ref] = usePlane<THREE.Mesh>(() => ({
    rotation: [-Math.PI / 2, 0, 0],
    position,
  }));
  return (
    <mesh ref={ref as React.Ref<THREE.Mesh>} receiveShadow>
      <planeGeometry args={[10, 10]} />
      <meshStandardMaterial color="#808080" side={THREE.DoubleSide} />
    </mesh>
  );
};

interface ColoredBoxProps {
  position: [number, number, number];
  color: string;
}

const ColoredBox: React.FC<ColoredBoxProps> = ({ position, color }) => {
  const [ref] = useBox<THREE.Mesh>(() => ({
    mass: 0,
    position,
    args: [0.2, 0.2, 0.2],
  }));
  return (
    <mesh ref={ref} castShadow receiveShadow>
      <boxGeometry args={[0.2, 0.2, 0.2]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};

const RobotUpdater: React.FC<{
  controllerRef: React.MutableRefObject<RobotController | null>;
}> = ({ controllerRef }) => {
  useFrame(() => {
    if (controllerRef.current) controllerRef.current.update();
  });
  return null;
};

const SceneSetter: React.FC<{ setScene: (scene: THREE.Scene) => void }> = ({ setScene }) => {
  const { scene } = useThree();
  useEffect(() => {
    setScene(scene);
  }, [scene, setScene]);
  return null;
};

const ForearmView: React.FC<{
  scene: THREE.Scene;
  robotCamera: THREE.PerspectiveCamera;
}> = ({ scene, robotCamera }) => {
  const viewRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  useEffect(() => {
    if (!viewRef.current) return;
    viewRef.current.innerHTML = "";
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(208, 117);
    renderer.setClearColor(0x000000);
    renderer.shadowMap.enabled = true;
    viewRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    const animate = () => {
      renderer.render(scene, robotCamera);
      requestAnimationFrame(animate);
    };
    animate();
    return () => {
      renderer.dispose();
    };
  }, [scene, robotCamera]);
  return <div ref={viewRef} style={{ width: "100%", height: "100%" }} />;
};

const Home: NextPage = () => {
  const [isClient, setIsClient] = useState(false);
  const [lastContent, setLastContent] = useState<string>("");
  const robotRef = useRef<RobotParts>(null);
  const [demoState, setDemoState] = useState(false);
  const [command, setCommand] = useState("");
  const [forearmCamera, setForearmCamera] = useState<THREE.PerspectiveCamera | null>(null);
  const [scene, setScene] = useState<THREE.Scene | null>(null);
  const robotControllerRef = useRef<RobotController | null>(null);
  const initialLoadRef = useRef(true);
  useEffect(() => {
    setIsClient(true);
  }, []);
  useEffect(() => {
    const fetchAndProcessFile = async () => {
      try {
        const response = await fetch("/output/output.txt");
        const text = await response.text();
        if (initialLoadRef.current) {
          initialLoadRef.current = false;
          setLastContent(text);
          return;
        }
        if (text !== lastContent && robotControllerRef.current) {
          const controller = robotControllerRef.current;
          controller.resetToStart();
          const commands = text.split(", ").map((cmd) => cmd.trim());
          commands.forEach((cmd) => {
            if (cmd === "C") return;
            const match = cmd.match(/^([BSE])([+-]\d+)$/i);
            if (match) {
              const part = match[1].toUpperCase();
              const deltaRad = THREE.MathUtils.degToRad(parseInt(match[2]));
              controller.addCommand(part, deltaRad);
            }
          });
          setLastContent(text);
        }
      } catch (error) {
        console.error("Error reading output.txt:", error);
      }
    };
    fetchAndProcessFile();
    const interval = setInterval(fetchAndProcessFile, 1000);
    return () => clearInterval(interval);
  }, [lastContent]);
  useEffect(() => {
    const interval = setInterval(() => {
      if (robotRef.current && robotRef.current.camera) {
        setForearmCamera(robotRef.current.camera);
        clearInterval(interval);
      }
    }, 100);
    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    let active = true;
    const checkRobot = () => {
      if (active && robotRef.current && !robotControllerRef.current) {
        robotControllerRef.current = new RobotController(robotRef.current);
      }
      if (active && !robotControllerRef.current) {
        requestAnimationFrame(checkRobot);
      }
    };
    requestAnimationFrame(checkRobot);
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!robotControllerRef.current) return;
      const controller = robotControllerRef.current;
      const key = e.key.toLowerCase();
      if (key === "m") {
        controller.demoMode = !controller.demoMode;
        setDemoState(controller.demoMode);
        return;
      }
      if (controller.demoMode) return;
      switch (key) {
        case "q":
          controller.rotateBase(1);
          break;
        case "e":
          controller.rotateBase(-1);
          break;
        case "w":
          controller.moveShoulder(-1);
          break;
        case "s":
          controller.moveShoulder(1);
          break;
        case "r":
          controller.moveElbow(-1);
          break;
        case "f":
          controller.moveElbow(1);
          break;
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);
  const handleCommandSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!robotControllerRef.current) return;
    const controller = robotControllerRef.current;
    const cmd = command.trim();
    const match = cmd.match(/^([BSE])([+-]\d+(?:\.\d+)?)$/i);
    if (match) {
      const part = match[1].toUpperCase();
      const deltaRad = THREE.MathUtils.degToRad(parseFloat(match[2]));
      controller.addCommand(part, deltaRad);
    } else if (cmd.toLowerCase() === "m") {
      controller.demoMode = !controller.demoMode;
      setDemoState(controller.demoMode);
    }
    setCommand("");
  };
  return (
    <>
      <Head>
        <title>Robot Arm</title>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <style>{`
            body, html {
              margin: 0;
              padding: 0;
              width: 100%;
              height: 100%;
            }
            canvas {
              display: block;
            }
            #commandForm {
              position: fixed;
              bottom: 20px;
              left: 50%;
              transform: translateX(-50%);
              z-index: 10;
            }
            #commandForm input {
              padding: 10px;
              line-height: 1;
              font-size: 12px;
              width: 300px;
              border: 0;
              color: #000;
              background: #fff;
            }
            #forearmViewContainer {
              position: fixed;
              top: 10px;
              right: 10px;
              z-index: 20;
              width: 210px;
              height: 119px;
              box-sizing: border-box;
              border: 1px solid #fff;
              background: #000;
            }
            #instructions {
              position: absolute;
              top: 20px;
              left: 20px;
              z-index: 10;
              color: #fff;
              font-family: Arial, sans-serif;
              line-height: 1.5;
              font-size: 12px;
            }
            #instructions ul {
              margin: 0;
            }
            #instructions li {
              margin-bottom: 5px;
            }
          `}</style>
      </Head>
      {(!scene || !forearmCamera) && (
        <div style={{ position: "absolute", top: 0, left: 0, width: "100vw", height: "100vh", background: "#000", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000, color: "#fff", fontSize: "24px" }}>
          Loading...
        </div>
      )}
      {isClient && (
        <Canvas
          shadows
          camera={{ position: [2, 1, 5], fov: 75 }}
          style={{ width: "100vw", height: "100vh" }}
        >
          <SceneSetter setScene={setScene} />
          <color attach="background" args={["#000"]} />
          <ambientLight intensity={0.4} />
          <directionalLight
            position={[0, 10, 10]}
            intensity={0.5}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <OrbitControls maxPolarAngle={Math.PI / 2 - 0.1} minPolarAngle={0} />
          <Grid
            position={[0, 0.001, 0]}
            args={[10, 10]}
            cellColor="#bbbbbb"
            sectionColor="#dddddd"
            cellSize={1}
            sectionSize={10}
            fadeDistance={20}
            fadeStrength={1}
          />
          <Physics gravity={[0, -9.82, 0]}>
            <RobotUpdater controllerRef={robotControllerRef} />
            <RobotArm ref={robotRef} position={[0, 0.11, 0]} />
            <Floor position={[0, 0, 0]} />
            <ColoredBox position={[1, 0.1, 0]} color="red" />
            <ColoredBox position={[0, 0.1, 1]} color="green" />
            <ColoredBox position={[-1, 0.1, 0]} color="blue" />
          </Physics>
        </Canvas>
      )}
      <div id="forearmViewContainer">
        {scene && forearmCamera && <ForearmView scene={scene} robotCamera={forearmCamera} />}
      </div>
      <div id="commandForm">
        <form onSubmit={handleCommandSubmit}>
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="Enter command, e.g. B+30, S-15, E+20, or M"
            suppressHydrationWarning={true}
          />
        </form>
      </div>
    </>
  );
};

export default Home;
