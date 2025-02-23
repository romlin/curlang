import {NextPage} from 'next'
import React, {
    useEffect,
    useRef,
    useState,
    forwardRef,
    useImperativeHandle,
    memo
} from 'react'
import Head from 'next/head'
import {Canvas, useFrame, useThree} from '@react-three/fiber'
import {OrbitControls, Grid} from '@react-three/drei'
import {Physics, usePlane, useBox} from '@react-three/cannon'
import * as THREE from 'three'

interface RobotParts {
    base?: THREE.Object3D
    shoulder?: THREE.Object3D
    upperArm?: THREE.Object3D
    elbow?: THREE.Object3D
    forearm?: THREE.Object3D
    pincerBase?: THREE.Object3D
    pincerClaw1?: THREE.Object3D
    pincerClaw2?: THREE.Object3D
    camera?: THREE.PerspectiveCamera
}

class RobotController {
    robotParts: RobotParts
    demoMode = false
    targetBaseRotation = 0
    targetShoulderRotation = 0
    targetElbowRotation = 0
    isBaseMoving = false
    isShoulderMoving = false
    isElbowMoving = false
    ROT_SPEED = 0.1
    LERP_FACTOR = 0.05
    shoulderMin = THREE.MathUtils.degToRad(0)
    shoulderMax = THREE.MathUtils.degToRad(45)
    elbowMin = THREE.MathUtils.degToRad(-90)
    elbowMax = THREE.MathUtils.degToRad(45)

    constructor(robotParts: RobotParts) {
        this.robotParts = robotParts
    }

    rotateBase(direction: number) {
        if (!this.robotParts.base) return
        this.robotParts.base.rotation.y += this.ROT_SPEED * direction
    }

    moveShoulder(direction: number) {
        if (!this.robotParts.shoulder) return
        const delta = this.ROT_SPEED * direction
        this.robotParts.shoulder.rotation.x = THREE.MathUtils.clamp(
            this.robotParts.shoulder.rotation.x + delta,
            this.shoulderMin,
            this.shoulderMax
        )
    }

    moveElbow(direction: number) {
        if (!this.robotParts.elbow) return
        const delta = this.ROT_SPEED * direction
        this.robotParts.elbow.rotation.x = THREE.MathUtils.clamp(
            this.robotParts.elbow.rotation.x + delta,
            this.elbowMin,
            this.elbowMax
        )
    }

    update() {
        const {base, shoulder, elbow} = this.robotParts
        if (this.isBaseMoving && base) {
            base.rotation.y += (this.targetBaseRotation - base.rotation.y) * this.LERP_FACTOR
            if (Math.abs(base.rotation.y - this.targetBaseRotation) < 0.01) {
                this.isBaseMoving = false
            }
        }
        if (this.isShoulderMoving && shoulder) {
            shoulder.rotation.x += (this.targetShoulderRotation - shoulder.rotation.x) * this.LERP_FACTOR
            if (Math.abs(shoulder.rotation.x - this.targetShoulderRotation) < 0.01) {
                this.isShoulderMoving = false
            }
        }
        if (this.isElbowMoving && elbow) {
            elbow.rotation.x += (this.targetElbowRotation - elbow.rotation.x) * this.LERP_FACTOR
            if (Math.abs(elbow.rotation.x - this.targetElbowRotation) < 0.01) {
                this.isElbowMoving = false
            }
        }
        if (this.demoMode) this.rotateBase(1)
    }
}

interface RobotArmProps {
    clawClosed: boolean
}

function RobotArmCollider({parentRef}: {
    parentRef: React.RefObject<THREE.Group>
}) {
    const [colliderRef] = useBox(() => ({
        mass: 0,
        args: [0.5, 0.3, 0.5],
        position: [0, 0, 0],
    }))
    useFrame(() => {
        if (parentRef.current && colliderRef.current) {
            const pos = new THREE.Vector3()
            parentRef.current.getWorldPosition(pos)
            colliderRef.current.position.copy(pos)
            const quat = new THREE.Quaternion()
            parentRef.current.getWorldQuaternion(quat)
            colliderRef.current.quaternion.copy(quat)
        }
    })
    return <mesh ref={colliderRef} visible={false}/>
}

const RobotArm = memo(
    forwardRef<RobotParts, RobotArmProps>(({clawClosed}, ref) => {
        const baseRef = useRef<THREE.Group>(null)
        const shoulderRef = useRef<THREE.Group>(null)
        const upperArmRef = useRef<THREE.Group>(null)
        const elbowRef = useRef<THREE.Group>(null)
        const forearmRef = useRef<THREE.Group>(null)
        const pincerBaseRef = useRef<THREE.Group>(null)
        const pincerClaw1Ref = useRef<THREE.Mesh>(null)
        const pincerClaw2Ref = useRef<THREE.Mesh>(null)
        const forearmCameraRef = useRef<THREE.PerspectiveCamera>(null)
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
        }))
        return (
            <group position={[0, 0.1, 0]}>
                <group ref={baseRef}>
                    <mesh castShadow>
                        <cylinderGeometry args={[0.2, 0.2, 0.2, 32]}/>
                        <meshNormalMaterial/>
                    </mesh>
                    <RobotArmCollider parentRef={baseRef}/>
                    <group ref={shoulderRef} position={[0, 0.1, 0]}>
                        <mesh castShadow>
                            <sphereGeometry args={[0.2, 32, 32]}/>
                            <meshNormalMaterial/>
                        </mesh>
                        <group ref={upperArmRef} position={[0, 0.5, 0]}>
                            <mesh castShadow>
                                <boxGeometry args={[0.1, 1, 0.1]}/>
                                <meshNormalMaterial/>
                            </mesh>
                            <group
                                ref={elbowRef}
                                position={[0, 0.6, 0]}
                                rotation={[THREE.MathUtils.degToRad(45), 0, Math.PI / 2]}>
                                <mesh castShadow>
                                    <cylinderGeometry
                                        args={[0.15, 0.15, 0.15, 32]}/>
                                    <meshNormalMaterial/>
                                </mesh>
                                <group ref={forearmRef} position={[0, 0, 0.4]}>
                                    <mesh castShadow>
                                        <boxGeometry args={[0.1, 0.1, 0.7]}/>
                                        <meshNormalMaterial/>
                                    </mesh>
                                    <group ref={pincerBaseRef}
                                           position={[0, 0, 0.36]}>
                                        <mesh castShadow>
                                            <boxGeometry
                                                args={[0.2, 0.4, 0.05]}/>
                                            <meshNormalMaterial/>
                                        </mesh>
                                        <mesh
                                            ref={pincerClaw1Ref}
                                            position={[0, clawClosed ? 0.05 : 0.175, 0.075]}
                                            castShadow>
                                            <boxGeometry
                                                args={[0.2, 0.05, 0.1]}/>
                                            <meshNormalMaterial/>
                                        </mesh>
                                        <mesh
                                            ref={pincerClaw2Ref}
                                            position={[0, clawClosed ? -0.05 : -0.175, 0.075]}
                                            castShadow>
                                            <boxGeometry
                                                args={[0.2, 0.05, 0.1]}/>
                                            <meshNormalMaterial/>
                                        </mesh>
                                    </group>
                                    <perspectiveCamera
                                        ref={forearmCameraRef}
                                        fov={60}
                                        aspect={220 / 140}
                                        near={0.1}
                                        far={100}
                                        position={[0.15, 0, 0.25]}
                                        rotation={[0, Math.PI, Math.PI / 2]}
                                    />
                                </group>
                            </group>
                        </group>
                    </group>
                </group>
            </group>
        )
    })
)
RobotArm.displayName = 'RobotArm'

function Floor(props: JSX.IntrinsicElements['mesh']) {
    const [ref] = usePlane(() => ({rotation: [-Math.PI / 2, 0, 0], ...props}))
    return (
        <mesh ref={ref} receiveShadow>
            <planeGeometry args={[10, 10]}/>
            <meshStandardMaterial color="#808080" side={THREE.DoubleSide}/>
        </mesh>
    )
}

function BoxUpdater({
                        robotRef,
                        boxRef,
                        isBoxGrabbed,
                    }: {
    robotRef: React.MutableRefObject<RobotParts | null>
    boxRef: React.MutableRefObject<any>
    isBoxGrabbed: boolean
}) {
    useFrame(() => {
        if (isBoxGrabbed && robotRef.current && boxRef.current) {
            const pincer = robotRef.current.pincerBase
            const pincerWorldPos = new THREE.Vector3()
            pincer.getWorldPosition(pincerWorldPos)
            const pincerQuat = new THREE.Quaternion()
            pincer.getWorldQuaternion(pincerQuat)
            const offset = new THREE.Vector3(0, 0, 0.3)
            offset.applyQuaternion(pincerQuat)
            const newPos = pincerWorldPos.clone().add(offset)
            boxRef.current.api.position.set(newPos.x, newPos.y, newPos.z)
            boxRef.current.api.quaternion.set(
                pincerQuat.x,
                pincerQuat.y,
                pincerQuat.z,
                pincerQuat.w
            )
        }
    })
    return null
}

interface InteractiveBoxProps {
    isBoxGrabbed: boolean
}

const InteractiveBox = forwardRef<any, InteractiveBoxProps>(
    ({isBoxGrabbed}, ref) => {
        const [boxRef, api] = useBox(() => ({
            mass: 1,
            position: [1, 0.15, 0],
            args: [0.3, 0.3, 0.3],
        }));
        useImperativeHandle(ref, () => ({mesh: boxRef, api}), [boxRef, api]);

        // Update the body type and mass based on isBoxGrabbed
        useEffect(() => {
            if (boxRef.current && boxRef.current.cannonBody) {
                const body = boxRef.current.cannonBody;
                if (isBoxGrabbed) {
                    body.type = 2; // Body.KINEMATIC
                    body.mass = 0;
                } else {
                    body.type = 1; // Body.DYNAMIC
                    body.mass = 1;
                }
                body.updateMassProperties();
            }
        }, [isBoxGrabbed]);

        return (
            <mesh ref={boxRef} castShadow>
                <boxGeometry args={[0.3, 0.3, 0.3]}/>
                <meshStandardMaterial
                    color={isBoxGrabbed ? "red" : "lightgrey"}/>
            </mesh>
        );
    }
);
InteractiveBox.displayName = 'InteractiveBox';

function RobotUpdater() {
    useFrame(() => {
        if (robotControllerRef.current) robotControllerRef.current.update()
    })
    return null
}

const robotControllerRef = {current: null as RobotController | null}

function SceneSetter({setScene}: { setScene: (scene: THREE.Scene) => void }) {
    const {scene} = useThree()
    useEffect(() => {
        setScene(scene)
    }, [scene, setScene])
    return null
}

function ForearmView({
                         scene,
                         robotCamera,
                     }: {
    scene: THREE.Scene
    robotCamera: THREE.PerspectiveCamera
}) {
    const viewRef = useRef<HTMLDivElement>(null)
    const rendererRef = useRef<THREE.WebGLRenderer>()
    useEffect(() => {
        if (!viewRef.current) return
        viewRef.current.innerHTML = ''
        const renderer = new THREE.WebGLRenderer({antialias: true})
        renderer.setSize(218, 138)
        renderer.setClearColor(0x000000)
        viewRef.current.appendChild(renderer.domElement)
        rendererRef.current = renderer
        const animate = () => {
            renderer.render(scene, robotCamera)
            requestAnimationFrame(animate)
        }
        animate()
        return () => {
            renderer.dispose()
        }
    }, [scene, robotCamera])
    return (
        <div
            ref={viewRef}
            style={{
                width: '220px',
                height: '140px',
                border: '1px solid #ffffff'
            }}
        />
    )
}

function ProcessOutput({
                           runCommands,
                       }: {
    runCommands: (commands: string[]) => void
}) {
    const [lastProcessed, setLastProcessed] = useState('')
    const isInitialLoad = useRef(true)
    useEffect(() => {
        const fetchContent = async () => {
            try {
                const res = await fetch('/output/output.txt', {cache: 'no-cache'})
                const text = await res.text()
                if (isInitialLoad.current) {
                    isInitialLoad.current = false
                    setLastProcessed(text)
                    return
                }
                if (text && text.trim() && text !== lastProcessed) {
                    setLastProcessed(text)
                    const commands = text
                        .split(',')
                        .map(cmd => cmd.trim())
                        .filter(cmd => cmd)
                    if (commands.length > 0) {
                        runCommands(commands)
                    }
                }
            } catch (error) {
                console.error('Error fetching output.txt', error)
            }
        }
        fetchContent()
        const interval = setInterval(fetchContent, 2000)
        return () => clearInterval(interval)
    }, [lastProcessed, runCommands])
    return null
}

const Home: NextPage = () => {
    const robotRef = useRef<RobotParts>(null)
    const boxRef = useRef<any>(null)
    const [demoState, setDemoState] = useState(false)
    const [command, setCommand] = useState('')
    const [isClawClosed, setIsClawClosed] = useState(false)
    const [isBoxGrabbed, setIsBoxGrabbed] = useState(false)
    const [forearmCamera, setForearmCamera] = useState<THREE.PerspectiveCamera | null>(null)
    const [scene, setScene] = useState<THREE.Scene | null>(null)
    const autoRunningRef = useRef(false)
    const runCommandsSequence = (commands: string[]) => {
        autoRunningRef.current = true;
        let index = 0;
        const executeNext = () => {
            if (index >= commands.length) {
                autoRunningRef.current = false;
                return;
            }
            const cmd = commands[index].trim();

            // Check for the "C" command (toggle claw)
            if (cmd.toLowerCase() === 'c') {
                setIsClawClosed(prev => {
                    const newState = !prev;
                    if (newState && robotRef.current && boxRef.current) {
                        const pincerPos = new THREE.Vector3();
                        robotRef.current.pincerBase.getWorldPosition(pincerPos);
                        const boxPos = new THREE.Vector3();
                        boxRef.current.mesh.current.getWorldPosition(boxPos);
                        if (pincerPos.distanceTo(boxPos) < 0.5) {
                            setIsBoxGrabbed(true);
                            boxRef.current.api.mass.set(0);
                        }
                    } else if (!newState && boxRef.current) {
                        setIsBoxGrabbed(false);
                        boxRef.current.api.mass.set(1);
                    }
                    return newState;
                });
            } else if (cmd.toLowerCase() === 'm') {
                // Toggle demo mode
                if (robotControllerRef.current) {
                    robotControllerRef.current.demoMode = !robotControllerRef.current.demoMode;
                    setDemoState(robotControllerRef.current.demoMode);
                }
            } else {
                // Process commands for B, S, and E using a regex
                const match = cmd.match(/^([BSE])([+-]\d+(?:\.\d+)?)$/i);
                if (match && robotControllerRef.current) {
                    const part = match[1].toUpperCase();
                    const deltaRad = THREE.MathUtils.degToRad(parseFloat(match[2]));
                    switch (part) {
                        case 'B': {
                            const currentRotation = robotControllerRef.current.robotParts.base?.rotation.y || 0;
                            robotControllerRef.current.targetBaseRotation = currentRotation + deltaRad;
                            robotControllerRef.current.isBaseMoving = true;
                            break;
                        }
                        case 'S': {
                            const currentRotation = robotControllerRef.current.robotParts.shoulder?.rotation.x || 0;
                            const newRotation = THREE.MathUtils.clamp(
                                currentRotation + deltaRad,
                                robotControllerRef.current.shoulderMin,
                                robotControllerRef.current.shoulderMax
                            );
                            robotControllerRef.current.targetShoulderRotation = newRotation;
                            robotControllerRef.current.isShoulderMoving = true;
                            break;
                        }
                        case 'E': {
                            const currentRotation = robotControllerRef.current.robotParts.elbow?.rotation.x || 0;
                            const newRotation = THREE.MathUtils.clamp(
                                currentRotation + deltaRad,
                                robotControllerRef.current.elbowMin,
                                robotControllerRef.current.elbowMax
                            );
                            robotControllerRef.current.targetElbowRotation = newRotation;
                            robotControllerRef.current.isElbowMoving = true;
                            break;
                        }
                        default:
                            console.log('Invalid command: ' + cmd);
                    }
                } else {
                    console.log('Invalid command format: ' + cmd);
                }
            }
            index++;
            setTimeout(executeNext, 500);
        };
        executeNext();
    };
    useEffect(() => {
        const interval = setInterval(() => {
            if (robotRef.current && robotRef.current.camera) {
                setForearmCamera(robotRef.current.camera)
                clearInterval(interval)
            }
        }, 100)
        return () => clearInterval(interval)
    }, [])
    useEffect(() => {
        let active = true

        function checkRobot() {
            if (active && robotRef.current && !robotControllerRef.current) {
                robotControllerRef.current = new RobotController(robotRef.current)
            }
            if (active && !robotControllerRef.current) {
                requestAnimationFrame(checkRobot)
            }
        }

        requestAnimationFrame(checkRobot)
        return () => {
            active = false
        }
    }, [])
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (autoRunningRef.current) return
            if (!robotControllerRef.current) return
            const key = e.key.toLowerCase()
            if (key === 'm') {
                robotControllerRef.current.demoMode = !robotControllerRef.current.demoMode
                setDemoState(robotControllerRef.current.demoMode)
                return
            }
            if (key === 'c') {
                setIsClawClosed(prev => {
                    const newState = !prev
                    if (newState && robotRef.current && boxRef.current) {
                        const pincerPos = new THREE.Vector3()
                        robotRef.current.pincerBase.getWorldPosition(pincerPos)
                        const boxPos = new THREE.Vector3()
                        boxRef.current.mesh.current.getWorldPosition(boxPos)
                        if (pincerPos.distanceTo(boxPos) < 0.5) {
                            setIsBoxGrabbed(true)
                            boxRef.current.api.mass.set(0)
                        }
                    } else if (!newState && boxRef.current) {
                        setIsBoxGrabbed(false)
                        boxRef.current.api.mass.set(1)
                    }
                    return newState
                })
                return
            }
            if (robotControllerRef.current.demoMode) return
            switch (key) {
                case 'q':
                    robotControllerRef.current.rotateBase(1)
                    break
                case 'e':
                    robotControllerRef.current.rotateBase(-1)
                    break
                case 'w':
                    robotControllerRef.current.moveShoulder(-1)
                    break
                case 's':
                    robotControllerRef.current.moveShoulder(1)
                    break
                case 'r':
                    robotControllerRef.current.moveElbow(-1)
                    break
                case 'f':
                    robotControllerRef.current.moveElbow(1)
                    break
                default:
                    break
            }
        }
        window.addEventListener('keydown', handleKeyDown)
        return () => window.removeEventListener('keydown', handleKeyDown)
    }, [isBoxGrabbed])
    const handleCommandSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault()
        if (autoRunningRef.current) return
        if (!robotControllerRef.current) return
        const cmd = command.trim()
        const match = cmd.match(/^([BSE])([+-]\d+(?:\.\d+)?)$/i)
        if (match) {
            const part = match[1].toUpperCase()
            const deltaRad = THREE.MathUtils.degToRad(parseFloat(match[2]))
            switch (part) {
                case 'B': {
                    const currentRotation = robotControllerRef.current.robotParts.base?.rotation.y || 0
                    robotControllerRef.current.targetBaseRotation = currentRotation + deltaRad
                    robotControllerRef.current.isBaseMoving = true
                    break
                }
                case 'S': {
                    const currentRotation = robotControllerRef.current.robotParts.shoulder?.rotation.x || 0
                    const newRotation = THREE.MathUtils.clamp(
                        currentRotation + deltaRad,
                        robotControllerRef.current.shoulderMin,
                        robotControllerRef.current.shoulderMax
                    )
                    robotControllerRef.current.targetShoulderRotation = newRotation
                    robotControllerRef.current.isShoulderMoving = true
                    break
                }
                case 'E': {
                    const currentRotation = robotControllerRef.current.robotParts.elbow?.rotation.x || 0
                    const newRotation = THREE.MathUtils.clamp(
                        currentRotation + deltaRad,
                        robotControllerRef.current.elbowMin,
                        robotControllerRef.current.elbowMax
                    )
                    robotControllerRef.current.targetElbowRotation = newRotation
                    robotControllerRef.current.isElbowMoving = true
                    break
                }
                default:
                    console.log('Invalid command: ' + cmd)
            }
        } else if (cmd.toLowerCase() === 'm') {
            robotControllerRef.current.demoMode = !robotControllerRef.current.demoMode
            setDemoState(robotControllerRef.current.demoMode)
        } else if (cmd.toLowerCase() === 'c') {
            setIsClawClosed(prev => {
                const newState = !prev
                if (newState && robotRef.current && boxRef.current) {
                    const pincerPos = new THREE.Vector3()
                    robotRef.current.pincerBase.getWorldPosition(pincerPos)
                    const boxPos = new THREE.Vector3()
                    boxRef.current.mesh.current.getWorldPosition(boxPos)
                    if (pincerPos.distanceTo(boxPos) < 0.5) {
                        setIsBoxGrabbed(true)
                        boxRef.current.api.mass.set(0)
                    }
                } else if (!newState && boxRef.current) {
                    setIsBoxGrabbed(false)
                    boxRef.current.api.mass.set(1)
                }
                return newState
            })
        } else {
            console.log('Invalid command: ' + cmd)
        }
        setCommand('')
    }
    return (
        <>
            <Head>
                <title>Robot Arm with Forearm View and Interactive Box</title>
                <meta charSet="UTF-8"/>
                <meta name="viewport"
                      content="width=device-width, initial-scale=1.0"/>
                <style>{`
          body, html { margin: 0; padding: 0; width: 100%; height: 100%; }
          canvas { display: block; }
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
          #instructions ul { margin: 0; }
          #instructions li { margin-bottom: 5px; }
          #commandForm {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 10;
          }
          #commandForm input {
            padding: 10px;
            font-size: 12px;
            width: 300px;
            border: 0px;
            border-radius: 0px;
            color: #000;
            background: #ffffff;
          }
          #forearmViewContainer {
            position: fixed;
            top: 10px;
            right: 10px;
            z-index: 20;
            background: #000;
          }
        `}</style>
            </Head>
            <div id="instructions">
                <ul>
                    <li><strong>Q/E:</strong> Rotate Base</li>
                    <li><strong>W/S:</strong> Move Shoulder</li>
                    <li><strong>R/F:</strong> Move Elbow</li>
                    <li><strong>M:</strong> Demo Mode
                        ({demoState ? 'ON' : 'OFF'})
                    </li>
                    <li><strong>C:</strong> Claw (pick up/release)
                    </li>
                </ul>
            </div>
            <Canvas shadows camera={{position: [2, 1, 5], fov: 75}}
                    style={{width: '100vw', height: '100vh'}}>
                <SceneSetter setScene={setScene}/>
                <color attach="background" args={['#000']}/>
                <ambientLight intensity={0.4}/>
                <directionalLight position={[0, 10, 10]} intensity={0.5}
                                  castShadow shadow-mapSize-width={1024}
                                  shadow-mapSize-height={1024}/>
                <OrbitControls maxPolarAngle={Math.PI / 2 - 0.1}
                               minPolarAngle={0}/>
                <Grid position={[0, 0.001, 0]} args={[10, 10]}
                      cellColor="#bbbbbb" sectionColor="#dddddd" cellSize={1}
                      sectionSize={10}/>
                <Physics gravity={[0, -9.82, 0]}>
                    <RobotUpdater/>
                    <BoxUpdater robotRef={robotRef} boxRef={boxRef}
                                isBoxGrabbed={isBoxGrabbed}/>
                    <RobotArm ref={robotRef} clawClosed={isClawClosed}/>
                    <InteractiveBox ref={boxRef} isBoxGrabbed={isBoxGrabbed}/>
                    <Floor/>
                </Physics>
            </Canvas>
            <div id="forearmViewContainer">
                {scene && forearmCamera &&
                    <ForearmView scene={scene} robotCamera={forearmCamera}/>}
            </div>
            <div id="commandForm">
                <form onSubmit={handleCommandSubmit}>
                    <input
                        type="text"
                        value={command}
                        onChange={(e) => setCommand(e.target.value)}
                        placeholder="Enter command, e.g. B+30, S-15, E+20, M, or C"
                    />
                </form>
            </div>
            <ProcessOutput runCommands={runCommandsSequence}/>
        </>
    )
}

export default Home