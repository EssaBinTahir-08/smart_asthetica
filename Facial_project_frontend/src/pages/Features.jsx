import Navbar from "../components/Navbar";
import { motion } from "framer-motion";
import Tilt from "react-parallax-tilt";
import { ScanFace, Camera, Sparkles, Gauge, ShieldCheck, Stars } from "lucide-react";

export default function Features() {
  const features = [
    {
      icon: <ScanFace size={42} />,
      title: "AI Face Landmark Detection",
      text: "Next-gen precision scanning trained on medical-grade datasets.",
    },
    {
      icon: <Camera size={42} />,
      title: "Real-time Live Preview",
      text: "Ultra-smooth GPU rendering with natural-light simulation.",
    },
    {
      icon: <Stars size={42} />,
      title: "Botox & Filler Simulation",
      text: "Accurate aesthetic enhancements using morphing models.",
    },
    {
      icon: <Gauge size={42} />,
      title: "Smart Intensity Control",
      text: "Adaptive enhancement levels powered by regression AI.",
    },
    {
      icon: <ShieldCheck size={42} />,
      title: "Encrypted & HIPAA Secure",
      text: "End-to-end encryption with private inference options.",
    },
    {
      icon: <Sparkles size={42} />,
      title: "Ultra-HD Neural Rendering",
      text: "4K photorealistic upscaling using super-resolution AI.",
    },
  ];

  return (
    <>
      <Navbar />

      <div className="pt-28 min-h-screen relative overflow-hidden">

        <div className="absolute w-96 h-96 bg-[#c084fc55] blur-[120px] top-10 left-10 animate-float-slow"></div>
        <div className="absolute w-80 h-80 bg-[#8b5cf655] blur-[110px] bottom-10 right-10 animate-float-slow"></div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-20 px-6 relative z-10"
        >
          <span className="inline-flex items-center gap-2 px-6 py-2 mb-6 text-sm font-bold rounded-full bg-accent-gradient text-white shadow-lg shadow-purple-500/20 uppercase tracking-widest">
            <Stars className="w-5 h-5" />
            Cutting Edge Technology
          </span>
          <h1 className="text-6xl md:text-8xl font-black mb-6 text-gray-900 tracking-tighter">
            Next-Gen <span className="bg-accent-gradient bg-clip-text text-transparent">AI Engine</span>
          </h1>
          <p className="max-w-2xl mx-auto text-lg md:text-xl text-gray-600 font-medium leading-relaxed">
            Harnessing the power of proprietary neural networks to redefine the future of aesthetic medicine.
          </p>
        </motion.div>

        <div className="max-w-7xl mx-auto grid md:grid-cols-2 lg:grid-cols-3 gap-10 px-6 pb-32 relative z-10">
          {features.map((f, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
            >
              <Tilt
                tiltMaxAngleX={12}
                tiltMaxAngleY={12}
                perspective={1200}
                scale={1.05}
                transitionSpeed={400}
                glareEnable={true}
                glareMaxOpacity={0.2}
                glareColor="white"
                glarePosition="all"
                className="h-full rounded-[2.5rem] p-10 bg-white/70 backdrop-blur-2xl border border-purple-100 shadow-[0_10px_40px_-10px_rgba(168,85,247,0.1)] transition-all duration-500 hover:shadow-[0_25px_60px_-10px_rgba(168,85,247,0.3)] hover:border-purple-300 group relative overflow-hidden flex flex-col items-center text-center"
              >
                {/* Internal Glow */}
                <div className="absolute -top-10 -right-10 w-40 h-40 bg-purple-200/30 rounded-full blur-3xl -z-10 group-hover:bg-purple-300/50 transition-colors duration-700" />

                <div className="absolute inset-0 bg-gradient-to-br from-purple-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 -z-10"></div>

                <div className="w-20 h-20 mb-8 flex items-center justify-center rounded-2xl bg-white shadow-xl group-hover:bg-accent-gradient group-hover:text-white group-hover:scale-110 group-hover:rotate-6 transition-all duration-500 relative z-10 border border-purple-50">
                  <div className="text-purple-600 group-hover:text-white transition-colors duration-500">
                    {f.icon}
                  </div>
                </div>

                <h3 className="text-2xl font-black text-gray-900 mb-4 group-hover:bg-accent-gradient group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300 relative z-10 tracking-tight">
                  {f.title}
                </h3>

                <p className="text-gray-600 font-medium leading-relaxed text-[16px] relative z-10 mb-8 flex-grow">
                  {f.text}
                </p>

                <div className="w-16 h-1.5 bg-purple-100 rounded-full scale-x-50 opacity-50 group-hover:scale-x-100 group-hover:opacity-100 group-hover:bg-purple-500 transition-all duration-500 ease-out origin-center relative z-10" />

                {/* Animated Accent Line */}
                <div className="absolute bottom-0 left-0 w-full h-1.5 bg-accent-gradient scale-x-0 group-hover:scale-x-100 transition-transform duration-500 ease-out origin-left" />
              </Tilt>
            </motion.div>
          ))}
        </div>
      </div>
    </>
  );
}  