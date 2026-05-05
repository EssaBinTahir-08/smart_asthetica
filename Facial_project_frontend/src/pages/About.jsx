import React, { useEffect, useRef } from "react";
import Navbar from "../components/Navbar";
import { motion } from "framer-motion";
import Tilt from "react-parallax-tilt";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: 0.08 * i, duration: 0.6, ease: "easeOut" },
  }),
};

export default function About() {
  const pipelineRef = useRef(null);

  useEffect(() => {
    const el = pipelineRef.current;
    if (!el) return;

    function onMove(e) {
      const rect = el.getBoundingClientRect();
      const dx = (e.clientX - (rect.left + rect.width / 2)) / rect.width;
      const dy = (e.clientY - (rect.top + rect.height / 2)) / rect.height;

      el.style.transform = `translate3d(${dx * 8}px, ${dy * 5}px, 0)`;
    }

    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  return (
    <>
      <Navbar />

      <div className="pt-28 relative min-h-screen overflow-hidden">



        <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6">

          <section className="pt-4 pb-10 text-center">
            <motion.h1
              initial={{ opacity: 0, y: -30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7 }}
              className="text-3xl sm:text-4xl md:text-6xl font-extrabold leading-snug text-[#4b3f72]"
            >
              Aesthetic Simulation —{" "}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-[#8c5bff] via-[#b887ff] to-[#d7baff]">
                AI Powered & Futuristic
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35, duration: 0.6 }}
              className="mt-5 text-base sm:text-lg md:text-xl text-[#5f5f8f] max-w-3xl mx-auto leading-relaxed px-2"
            >
              SmartAesthetica harnesses cutting-edge AI, neural rendering, and clinically validated datasets to produce ultra-realistic facial simulations — instantly. Every transformation is performed securely on private servers, ensuring complete confidentiality and peace of mind.
            </motion.p>


            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-4"
            >
              {[
                {
                  title: "Advanced AI",
                  desc: "State-of-the-art neural networks for precise facial simulations.",
                  color: "from-purple-600 to-indigo-500",
                },
                {
                  title: "HIPAA Compliant",
                  desc: "Fully secure and encrypted workflows for complete privacy.",
                  color: "from-pink-500 to-purple-500",
                },
                {
                  title: "Instant Previews",
                  desc: "High-fidelity results in real-time (<3 seconds).",
                  color: "from-blue-500 to-indigo-600",
                },
                {
                  title: "Explainable AI",
                  desc: "Landmark-driven transformations that are easy to understand.",
                  color: "from-purple-500 to-pink-400",
                },
              ].map((f, i) => (
                <Tilt
                  key={i}
                  tiltMaxAngleX={12}
                  tiltMaxAngleY={12}
                  perspective={1200}
                  scale={1.05}
                  transitionSpeed={400}
                  glareEnable={true}
                  glareMaxOpacity={0.25}
                  glareColor="white"
                  glarePosition="all"
                  className="p-8 sm:p-10 rounded-[2.5rem] bg-white/70 backdrop-blur-2xl border border-purple-100 shadow-[0_10px_40px_-10px_rgba(168,85,247,0.1)] transition-all duration-500 group hover:shadow-[0_25px_60px_-10px_rgba(168,85,247,0.3)] hover:border-purple-300 relative overflow-hidden h-full flex flex-col"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-50/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 -z-10"></div>

                  <div className={`w-6 h-6 mb-6 rounded-full bg-gradient-to-tr ${f.color} shadow-lg group-hover:scale-125 transition-transform duration-500 relative z-10`} />

                  <h3 className="text-xl sm:text-2xl font-black text-gray-900 mb-3 group-hover:bg-accent-gradient group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300 relative z-10 tracking-tight">
                    {f.title}
                  </h3>

                  <p className="text-gray-600 font-medium leading-relaxed relative z-10 text-[15px] mb-8 flex-grow">
                    {f.desc}
                  </p>

                  {/* Animated Line */}
                  <div className="w-12 h-1.5 bg-purple-100 rounded-full scale-x-50 opacity-50 group-hover:scale-x-100 group-hover:opacity-100 group-hover:bg-purple-500 transition-all duration-500 ease-out origin-left relative z-10" />
                </Tilt>
              ))}
            </motion.div>
          </section>

          <section id="pipeline" className="py-20">
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-2xl sm:text-4xl font-black text-center mb-16 text-gray-900 tracking-tight"
            >
              The <span className="bg-accent-gradient bg-clip-text text-transparent">Neural Pipeline</span>
            </motion.h2>

            <div ref={pipelineRef} className="relative px-2">
              <div className="w-full grid md:grid-cols-2 lg:grid-cols-4 gap-8">
                {[
                  { title: "Capture", desc: "Client image & landmarks" },
                  { title: "Analyze", desc: "Anatomy & measurements" },
                  { title: "Simulate", desc: "Neural rendering & 3D" },
                  { title: "Explain", desc: "Landmark-based deltas" },
                ].map((step, i) => (
                  <Tilt
                    key={i}
                    tiltMaxAngleX={12}
                    tiltMaxAngleY={12}
                    perspective={1200}
                    scale={1.03}
                    transitionSpeed={400}
                    glareEnable={true}
                    glareMaxOpacity={0.2}
                    glareColor="white"
                    glarePosition="all"
                    className="bg-white/70 backdrop-blur-2xl p-8 rounded-[2rem] border border-purple-100 shadow-[0_10px_30px_-5px_rgba(168,85,247,0.05)] transition-all duration-500 group hover:shadow-[0_20px_50px_-10px_rgba(168,85,247,0.2)] hover:border-purple-300 relative overflow-hidden flex flex-col justify-between"
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-100/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 -z-10"></div>

                    <div className="flex items-center gap-5 relative z-10">
                      <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-white font-black text-xl bg-accent-gradient shadow-lg group-hover:scale-110 group-hover:rotate-6 transition-all duration-500">
                        {i + 1}
                      </div>

                      <div>
                        <div className="font-black text-gray-900 text-lg group-hover:bg-accent-gradient group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300 tracking-tight">{step.title}</div>
                        <div className="text-sm text-gray-500 font-bold group-hover:text-gray-700 transition-colors uppercase tracking-widest text-[10px]">{step.desc}</div>
                      </div>
                    </div>

                    <div className="mt-10 h-2 w-full bg-purple-50 rounded-full overflow-hidden relative z-10 border border-purple-100/50">
                      <motion.div
                        initial={{ x: "-100%" }}
                        whileInView={{ x: "0%" }}
                        transition={{ duration: 1.2, delay: 0.5 + i * 0.2 }}
                        className="h-full rounded-full bg-accent-gradient shadow-[0_0_10px_rgba(168,85,247,0.5)]"
                      />
                    </div>
                  </Tilt>
                ))}
              </div>
            </div>
          </section>

          <footer className="py-10 text-center text-sm text-[#8e85b7]">
            © {new Date().getFullYear()} SmartAesthetica — AI-Powered Aesthetic Previews
          </footer>
        </main>
      </div>
    </>
  );
}
