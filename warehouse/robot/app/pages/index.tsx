"use client";

import { useState, useEffect, useRef, useCallback, memo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { GridHelper, Mesh } from "three";

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

interface PendingUpdate {
  text: string;
  textTimestamp: number;
}

const MAX_TEXT_LENGTH = 128;

function Sphere({ audioLevel }: { audioLevel: number }) {
  const meshRef = useRef<Mesh | null>(null);
  useFrame(() => {
    if (meshRef.current) {
      const scale = 1 + audioLevel * 0.5;
      meshRef.current.scale.set(scale, scale, scale);
    }
  });
  return (
    <mesh ref={meshRef} position={[0, 1, 0]}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshStandardMaterial color="#ff69b4" />
    </mesh>
  );
}

const GridFloor = memo(() => {
  const grid = new GridHelper(20, 20, "#ffffff", "#444444");
  return <primitive object={grid} />;
});
GridFloor.displayName = "GridFloor";

function Scene({ audioLevel }: { audioLevel: number }) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 10, 7.5]} intensity={1} />
      <Sphere audioLevel={audioLevel} />
      <GridFloor />
    </>
  );
}

const Robot: React.FC = () => {
  const [currentText, setCurrentText] = useState("Loading...");
  const [pendingUpdate, setPendingUpdate] = useState<PendingUpdate | null>(null);
  const [isSoundOn, setIsSoundOn] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  interface AudioRefType {
    context: AudioContext;
    source: AudioBufferSourceNode;
    analyser: AnalyserNode;
  }
  const audioRef = useRef<AudioRefType | null>(null);
  const checkTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const isSoundOnRef = useRef(isSoundOn);
  const currentTextRef = useRef(currentText);

  useEffect(() => {
    isSoundOnRef.current = isSoundOn;
  }, [isSoundOn]);

  useEffect(() => {
    currentTextRef.current = currentText;
  }, [currentText]);

  const truncateText = useCallback((text: string, maxLength: number): string => {
    if (text.length <= maxLength) return text;
    let truncated = text.slice(0, maxLength);
    const lastSpace = truncated.lastIndexOf(" ");
    if (lastSpace > 0) {
      truncated = truncated.slice(0, lastSpace);
    }
    truncated = truncated.replace(/[\s!,.?:;]+$/, "");
    return truncated + "...";
  }, []);

  const checkForTextUpdates = useCallback(async () => {
    try {
      const response = await fetch(`/output/output.txt?t=${Date.now()}`, {
        cache: "no-store",
      });
      if (!response.ok) return;
      const newText = truncateText(await response.text(), MAX_TEXT_LENGTH);
      const textTimestamp = Date.parse(response.headers.get("last-modified") || "");
      if (newText !== currentTextRef.current && !isNaN(textTimestamp)) {
        setPendingUpdate({ text: newText, textTimestamp });
      }
    } catch {}
  }, [truncateText]);

  useEffect(() => {
    const loadInitialText = async () => {
      try {
        const response = await fetch(`/output/output.txt?t=${Date.now()}`, {
          cache: "no-store",
        });
        const text = response.ok ? await response.text() : "File not found.";
        setCurrentText(truncateText(text, MAX_TEXT_LENGTH));
      } catch {}
    };
    loadInitialText();
    const intervalId = setInterval(checkForTextUpdates, 3000);
    return () => {
      clearInterval(intervalId);
      if (checkTimeoutRef.current) clearTimeout(checkTimeoutRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [checkForTextUpdates, truncateText]);

  const playAudio = useCallback(
    async (newText: string) => {
      try {
        if (audioRef.current) {
          audioRef.current.source.stop();
          audioRef.current.context.close();
          audioRef.current = null;
        }
        const response = await fetch(`/output/output.wav?t=${Date.now()}`);
        if (!response.ok) return;
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error("Web Audio API not supported");
        const audioContext = new AudioContextClass();
        const audioData = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(audioData);
        const bufferSource = audioContext.createBufferSource();
        bufferSource.buffer = audioBuffer;
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        bufferSource.connect(analyser);
        analyser.connect(audioContext.destination);
        setCurrentText(newText);
        bufferSource.start(0);
        audioRef.current = {
          context: audioContext,
          source: bufferSource,
          analyser,
        };
        const animateSphere = () => {
          if (!audioRef.current) return;
          audioRef.current.analyser.getByteFrequencyData(dataArray);
          const avg = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
          const normalized = avg / 255;
          setAudioLevel(normalized);
          animationFrameRef.current = requestAnimationFrame(animateSphere);
        };
        animateSphere();
        bufferSource.onended = () => {
          if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
          }
          setAudioLevel(0);
          audioRef.current = null;
          checkForTextUpdates();
        };
      } catch {
        setAudioLevel(0);
        audioRef.current = null;
        setPendingUpdate(null);
      }
    },
    [checkForTextUpdates]
  );

  useEffect(() => {
    if (!pendingUpdate) return;
    if (!isSoundOnRef.current) {
      setCurrentText(pendingUpdate.text);
      setPendingUpdate(null);
      return;
    }
    const verifyAndPlayAudio = async () => {
      try {
        const headResponse = await fetch(`/output/output.wav?t=${Date.now()}`, {
          method: "HEAD",
          cache: "no-store",
        });
        if (!headResponse.ok) throw new Error("Audio not found");
        const audioTimestamp = Date.parse(headResponse.headers.get("last-modified") || "");
        if (isNaN(audioTimestamp) || audioTimestamp <= pendingUpdate.textTimestamp) {
          throw new Error("Audio not updated yet");
        }
        await playAudio(pendingUpdate.text);
        setPendingUpdate(null);
      } catch {
        checkTimeoutRef.current = setTimeout(verifyAndPlayAudio, 1000);
      }
    };
    verifyAndPlayAudio();
    return () => {
      if (checkTimeoutRef.current) clearTimeout(checkTimeoutRef.current);
    };
  }, [pendingUpdate, playAudio]);

  const handleSoundToggle = () => {
    const newState = !isSoundOn;
    setIsSoundOn(newState);
    if (!newState && audioRef.current) {
      audioRef.current.source.stop();
      audioRef.current.context.close();
      audioRef.current = null;
      setAudioLevel(0);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 p-4">
      <button
        onClick={handleSoundToggle}
        className={`absolute top-4 right-4 cursor-pointer text-white ${isSoundOn ? "opacity-100" : "opacity-50"}`}
        title={isSoundOn ? "Turn sound off" : "Turn sound on"}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="w-8 h-8"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M11.25 5.25L6.75 9H3.75a.75.75 0 00-.75.75v4.5c0 .414.336.75.75.75h3l4.5 3.75v-12zM15.75 9a6 6 0 010 6m3-9a9 9 0 010 12"
          />
        </svg>
      </button>
      <div className="flex flex-col items-center space-y-4 w-full max-w-md">
        <div className="w-full h-64">
          <Canvas camera={{ position: [0, 5, 10], fov: 60 }}>
            <Scene audioLevel={audioLevel} />
            <OrbitControls />
          </Canvas>
        </div>
        <div className="w-full flex flex-col items-center">
          <p className="text-white font-sans text-sm leading-relaxed text-center">
            {currentText}
          </p>
          {pendingUpdate && isSoundOn && (
            <p className="text-sm leading-relaxed text-gray-400 mt-2">
              Generating speech...
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

Robot.displayName = "Robot";

export default Robot;