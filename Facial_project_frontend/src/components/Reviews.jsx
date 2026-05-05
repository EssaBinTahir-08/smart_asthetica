import { motion } from "framer-motion";

export default function Reviews() {
  const reviews = [
    {
      text: "The preview looked EXACTLY like my real post-treatment results. This saved me from unsure decisions.",
      name: "Emily Parker",
      role: "Verified Client",
    },
    {
      text: "My clinic uses SmartAesthetica daily. It reduces consultation time and improves patient confidence.",
      name: "Dr. Ayesha",
      role: "Cosmetic Surgeon",
    },
    {
      text: "Super clean UI, fast results, and extremely accurate face mapping. Highly recommended!",
      name: "John Miller",
      role: "User",
    },
  ];

  return (
    <section className="py-32 px-6 relative overflow-hidden bg-gradient-to-b from-[#f9f5ff] to-[#f1e8ff]">
      <div className="absolute top-[-120px] left-[-80px] w-[300px] h-[300px] bg-[#c8a7ff66] blur-[120px] rounded-full"></div>
      <div className="absolute bottom-[-150px] right-[-80px] w-[340px] h-[340px] bg-[#99d4ff66] blur-[140px] rounded-full"></div>

      <div className="text-center relative z-10 mb-16">
        <h2 className="text-5xl font-extrabold text-gray-900">
          User{" "}
          <span className="bg-gradient-to-r from-[#7f4bff] to-[#b88bff] text-transparent bg-clip-text">
            Reviews
          </span>
        </h2>
        <p className="text-gray-600 mt-3 text-lg">
          Real feedback. Real transformations.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-10 relative z-10">
        {reviews.map((review, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.2 }}
            whileHover={{
              y: -10,
              scale: 1.03,
              boxShadow:
                "0 25px 60px rgba(140,120,255,0.25), 0 0 35px rgba(150,110,255,0.25)",
            }}
            className="
              p-8 rounded-3xl 
              bg-white/60 backdrop-blur-2xl 
              border border-white/40 
              shadow-xl 
              transition-all duration-300 
              relative overflow-hidden group
            "
          >
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-all bg-gradient-to-br from-[#b77bff40] to-[#7bc7ff40] blur-[50px]"></div>

            <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
              <div className="absolute -top-1/2 left-0 w-full h-full bg-gradient-to-br from-white/20 to-transparent rotate-12 group-hover:translate-y-full transition-transform duration-1000"></div>
            </div>
            <p className="text-gray-700 italic relative z-10 leading-relaxed">
              “{review.text}”
            </p>
           
            <div className="mt-6 relative z-10">
              <h3 className="text-gray-900 font-semibold">{review.name}</h3>
              <p className="text-sm text-gray-500">{review.role}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
