import {NextPage} from 'next'
import React, {
    useEffect,
    useRef,
    useState,
    forwardRef,
    useImperativeHandle,
    memo,
} from 'react'
import Head from 'next/head'
import {Canvas, useFrame, useThree} from '@react-three/fiber'
import {OrbitControls, Grid} from '@react-three/drei'
import {Physics, usePlane} from '@react-three/cannon'
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

const RobotArm = memo(
    forwardRef<RobotParts>((_, ref) => {
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
                            <group ref={elbowRef} position={[0, 0.6, 0]}
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
                                        <mesh ref={pincerClaw1Ref}
                                              position={[0, 0.175, 0.075]}
                                              castShadow>
                                            <boxGeometry
                                                args={[0.2, 0.05, 0.1]}/>
                                            <meshNormalMaterial/>
                                        </mesh>
                                        <mesh ref={pincerClaw2Ref}
                                              position={[0, -0.175, 0.075]}
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
                                        position={[1, 0, 0.5]}
                                        rotation={
                                            [
                                                THREE.MathUtils.degToRad(0),
                                                THREE.MathUtils.degToRad(180),
                                                THREE.MathUtils.degToRad(90)
                                            ]
                                        }
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
    const [ref] = usePlane<THREE.Mesh>(() => ({rotation: [-Math.PI / 2, 0, 0], ...props}))
    return (
        <mesh ref={ref} receiveShadow>
            <planeGeometry args={[10, 10]}/>
            <meshStandardMaterial color="#808080" side={THREE.DoubleSide}/>
        </mesh>
    )
}

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

function ForearmView({scene, robotCamera}: {
    scene: THREE.Scene;
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
    return <div ref={viewRef} style={{
        width: '220px',
        height: '140px',
        border: '1px solid white'
    }}/>
}

const Home: NextPage = () => {
    const robotRef = useRef<RobotParts>(null)
    const [demoState, setDemoState] = useState(false)
    const [command, setCommand] = useState('')
    const [forearmCamera, setForearmCamera] = useState<THREE.PerspectiveCamera | null>(null)
    const [scene, setScene] = useState<THREE.Scene | null>(null)
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
            if (!robotControllerRef.current) return
            const key = e.key.toLowerCase()
            if (key === 'm') {
                robotControllerRef.current.demoMode = !robotControllerRef.current.demoMode
                setDemoState(robotControllerRef.current.demoMode)
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
    }, [])
    const handleCommandSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault()
        if (!robotControllerRef.current) return
        const cmd = command.trim()
        const match = cmd.match(/^([BSE])([+-]\d+(?:\.\d+)?)$/i)
        if (match) {
            const part = match[1].toUpperCase()
            const deltaRad = THREE.MathUtils.degToRad(parseFloat(match[2]))
            switch (part) {
                case 'B': {
                    robotControllerRef.current.targetBaseRotation =
                        robotControllerRef.current.robotParts.base!.rotation.y + deltaRad
                    robotControllerRef.current.isBaseMoving = true
                    break
                }
                case 'S': {
                    const currentRotation = robotControllerRef.current.robotParts.shoulder!.rotation.x
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
                    const currentRotation = robotControllerRef.current.robotParts.elbow!.rotation.x
                    const newRotation = THREE.MathUtils.clamp(
                        currentRotation + deltaRad,
                        robotControllerRef.current.elbowMin,
                        robotControllerRef.current.elbowMax
                    )
                    robotControllerRef.current.targetElbowRotation = newRotation
                    robotControllerRef.current.isElbowMoving = true
                    break
                }
            }
        } else if (cmd.toLowerCase() === 'm') {
            robotControllerRef.current.demoMode = !robotControllerRef.current.demoMode
            setDemoState(robotControllerRef.current.demoMode)
        } else {
            console.log('Invalid command: ' + cmd)
        }
        setCommand('')
    }
    return (
        <>
            <Head>
                <title>Robot Arm with Forearm View</title>
                <meta charSet="UTF-8"/>
                <meta name="viewport"
                      content="width=device-width, initial-scale=1.0"/>
                <style>{`
          body, html { margin: 0; padding: 0; width: 100%; height: 100%; }
          canvas { display: block; }
          #instructions { position: absolute; top: 20px; left: 20px; z-index: 10; color: #fff; font-family: Arial, sans-serif; line-height: 1.5; font-size: 12px; }
          #instructions ul { margin: 0; }
          #instructions li { margin-bottom: 5px; }
          #commandForm { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); z-index: 10; }
          #commandForm input { padding: 10px; font-size: 12px; width: 300px; border: 0; color: #000; }
          #forearmViewContainer { position: fixed; top: 10px; right: 10px; z-index: 20; background: #000; border: 1px solid #fff; }
        `}</style>
            </Head>
            <div id="instructions">
                <ul>
                    <li><strong>Q/E:</strong> Rotate Base (keyboard)</li>
                    <li><strong>W/S:</strong> Move Shoulder (keyboard)</li>
                    <li><strong>R/F:</strong> Move Elbow (keyboard)</li>
                    <li><strong>M:</strong> Toggle Demo Mode (keyboard/command)
                        (Current: {demoState ? 'ON' : 'OFF'})
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
                <Grid position={[0, 0.01, 0]} args={[10, 10]}
                      cellColor="#bbbbbb" sectionColor="#dddddd" cellSize={1}
                      sectionSize={10}/>
                <Physics gravity={[0, -9.82, 0]}>
                    <RobotUpdater/>
                    <RobotArm ref={robotRef}/>
                    <Floor/>
                </Physics>
            </Canvas>
            <div id="forearmViewContainer">
                {scene && forearmCamera &&
                    <ForearmView scene={scene} robotCamera={forearmCamera}/>}
            </div>
            <div id="commandForm">
                <form onSubmit={handleCommandSubmit}>
                    <input type="text" value={command}
                           onChange={(e) => setCommand(e.target.value)}
                           placeholder="Enter command, e.g. B+30, S-15, E+20, or M"/>
                </form>
            </div>
        </>
    )
}

export default Home